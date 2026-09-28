"""
Chat endpoints — RAG Q&A with citations, conversation management.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_db
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.message import Message
from app.models.user import User
from app.schemas.chat import (
    ChatAskRequest,
    ChatAskResponse,
    Citation,
    ConversationListResponse,
    ConversationResponse,
    MessageResponse,
)
from app.services import retrieval_service
from app.services.llm_service import get_llm_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/ask", response_model=ChatAskResponse, status_code=status.HTTP_200_OK)
def ask_question(
    request: ChatAskRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatAskResponse:
    """
    RAG Q&A: retrieve relevant context chunks, generate a grounded answer with citations,
    and persist conversation & message history.
    """
    # 1. Resolve or create Conversation
    if request.conversation_id:
        conversation = (
            db.query(Conversation)
            .filter(Conversation.id == request.conversation_id, Conversation.user_id == current_user.id)
            .first()
        )
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
    else:
        # Title snippet from query
        title = request.query[:50] + ("..." if len(request.query) > 50 else "")
        conversation = Conversation(
            user_id=current_user.id,
            title=title,
        )
        db.add(conversation)
        db.flush()

    # 2. Validate document_ids if provided (must belong to this user)
    valid_doc_ids: list[str] | None = None
    if request.document_ids:
        user_docs = (
            db.query(Document.id)
            .filter(Document.user_id == current_user.id, Document.id.in_(request.document_ids))
            .all()
        )
        valid_doc_ids = [d[0] for d in user_docs]
        if not valid_doc_ids:
            # None of the requested documents belong to this user
            context_chunks = []
        else:
            context_chunks = retrieval_service.retrieve_context(
                query=request.query,
                user_id=current_user.id,
                k=request.top_k,
                document_ids=valid_doc_ids,
            )
    else:
        context_chunks = retrieval_service.retrieve_context(
            query=request.query,
            user_id=current_user.id,
            k=request.top_k,
            document_ids=None,
        )

    # 3. Retrieve prior chat history (up to last 10 turns)
    prior_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
        .all()
    )
    chat_history = [
        {"role": m.role, "content": m.content}
        for m in prior_messages[-10:]
    ]

    # 4. Generate answer via LLM
    llm = get_llm_service()
    answer, raw_citations = llm.generate_rag_answer(
        query=request.query,
        context_chunks=context_chunks,
        chat_history=chat_history,
    )

    # 5. Persist user message & assistant message
    user_msg = Message(
        conversation_id=conversation.id,
        role="user",
        content=request.query,
        citations=[],
    )
    db.add(user_msg)

    asst_msg = Message(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        citations=raw_citations,
    )
    db.add(asst_msg)
    db.commit()

    citations = [
        Citation(
            chunk_id=c.get("chunk_id", ""),
            document_id=c.get("document_id", ""),
            text_snippet=c.get("text_snippet"),
        )
        for c in raw_citations
    ]

    return ChatAskResponse(
        conversation_id=conversation.id,
        message_id=asst_msg.id,
        answer=answer,
        citations=citations,
    )


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationListResponse:
    """List all conversations for the authenticated user."""
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    conv_responses = [
        ConversationResponse(
            id=c.id,
            user_id=c.user_id,
            title=c.title,
            created_at=c.created_at,
        )
        for c in conversations
    ]
    return ConversationListResponse(
        conversations=conv_responses,
        total=len(conv_responses),
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationResponse:
    """Get a conversation with all its messages in chronological order."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
        .all()
    )

    msg_responses = [
        MessageResponse(
            id=m.id,
            conversation_id=m.conversation_id,
            role=m.role,
            content=m.content,
            citations=m.citations,
            created_at=m.created_at,
        )
        for m in messages
    ]

    return ConversationResponse(
        id=conversation.id,
        user_id=conversation.user_id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=msg_responses,
    )


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a conversation and its messages."""
    conversation = (
        db.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == current_user.id)
        .first()
    )
    if not conversation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    db.delete(conversation)
    db.commit()
