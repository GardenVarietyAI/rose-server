"""Responses API schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class ResponsesContentItem(BaseModel):
    type: Literal["output_text"]
    text: str
    annotations: List[Any] = []


class ResponsesOutputItem(BaseModel):
    id: str
    type: Literal["message", "function_call"]
    status: Optional[str] = "completed"
    role: str
    content: Optional[List[ResponsesContentItem]] = None
    name: Optional[str] = None  # For function calls (deprecated)
    arguments: Optional[str] = None  # For function calls (deprecated)
    tool_calls: Optional[List[Dict[str, Any]]] = None  # OpenAI format tool calls


class ResponsesUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_tokens_details: Dict[str, int] = {"cached_tokens": 0}
    output_tokens_details: Dict[str, int] = {"reasoning_tokens": 0}


class ResponsesResponse(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: Literal["completed", "in_progress", "failed"] = "completed"
    output: List[ResponsesOutputItem]
    usage: ResponsesUsage


class ResponsesInputMessage(BaseModel):
    """Standard message input for responses API."""

    role: Literal["user", "assistant", "developer"] = Field(description="Message role")
    content: Union[str, List[Dict[str, Any]], None] = Field(description="Message content")


class ResponsesInputFunctionCall(BaseModel):
    """Function call message from assistant."""

    type: Literal["function_call"] = Field(description="Message type")
    id: str = Field(description="Function call ID")
    name: str = Field(description="Function name")
    arguments: Optional[str] = Field(description="Function arguments as JSON string")
    role: Literal["assistant"] = Field(default="assistant", description="Always assistant role")
    content: Union[str, None] = Field(default=None, description="Optional content")
    status: Optional[str] = Field(default=None, description="Call status")


class ResponsesInputFunctionOutput(BaseModel):
    """Function call output/result."""

    type: Literal["function_call_output"] = Field(description="Message type")
    call_id: Optional[str] = Field(description="ID of the function call this is responding to")
    output: str = Field(description="Function execution output")


ResponsesInput = Union[ResponsesInputMessage, ResponsesInputFunctionCall, ResponsesInputFunctionOutput]


class ResponsesRequest(BaseModel):
    """Responses API request format."""

    model: str = Field(description="Model to use for completion")
    input: Union[List[ResponsesInput], str] = Field(description="Input messages or text")
    modalities: List[Literal["text"]] = Field(default=["text"], description="Supported modalities")
    instructions: Optional[str] = Field(default=None, description="System instructions")
    stream: Optional[bool] = Field(default=False, description="Whether to stream the response")
    tools: Optional[List[Dict[str, Any]]] = Field(default=None, description="Available tools")
    tool_choice: Optional[str] = Field(default="auto", description="Tool choice strategy")
    parallel_tool_calls: Optional[bool] = Field(default=True, description="Whether to allow parallel tool calls")
    max_output_tokens: Optional[int] = Field(default=2048, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for sampling")
    top_p: Optional[float] = Field(default=0.8, description="Top-p value for sampling")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    store: Optional[bool] = Field(default=True, description="Whether to store the conversation")
    previous_response_id: Optional[str] = Field(default=None, description="Previous response ID")


class ResponseEventBase(BaseModel):
    """Base class for response events."""

    type: str = Field(description="Event type")
    sequence_number: int = Field(description="Sequence number for this event")


class ResponseCreatedEvent(ResponseEventBase):
    type: Literal["response.created"] = "response.created"
    response: Dict[str, Any] = Field(description="Response data")


class ResponseInProgressEvent(ResponseEventBase):
    type: Literal["response.in_progress"] = "response.in_progress"
    response: Dict[str, Any] = Field(description="Response data")


class ResponseContentPartAddedEvent(ResponseEventBase):
    type: Literal["response.content_part.added"] = "response.content_part.added"
    item_id: str = Field(description="Item ID")
    output_index: int = Field(description="Output index")
    content_index: int = Field(description="Content index")
    part: Dict[str, Any] = Field(description="Content part")


class ResponseOutputTextDeltaEvent(ResponseEventBase):
    type: Literal["response.output_text.delta"] = "response.output_text.delta"
    item_id: str = Field(description="Item ID")
    output_index: int = Field(description="Output index")
    content_index: int = Field(description="Content index")
    delta: str = Field(description="Text delta")


class ResponseOutputTextDoneEvent(ResponseEventBase):
    type: Literal["response.output_text.done"] = "response.output_text.done"
    item_id: str = Field(description="Item ID")
    output_index: int = Field(description="Output index")
    content_index: int = Field(description="Content index")
    text: str = Field(description="Complete text")


class ResponseContentPartDoneEvent(ResponseEventBase):
    type: Literal["response.content_part.done"] = "response.content_part.done"
    item_id: str = Field(description="Item ID")
    output_index: int = Field(description="Output index")
    content_index: int = Field(description="Content index")
    part: Dict[str, Any] = Field(description="Content part")


class ResponseOutputItemAddedEvent(ResponseEventBase):
    type: Literal["response.output_item.added"] = "response.output_item.added"
    output_index: int = Field(description="Output index")
    item: Dict[str, Any] = Field(description="Output item")


class ResponseOutputItemDoneEvent(ResponseEventBase):
    type: Literal["response.output_item.done"] = "response.output_item.done"
    output_index: int = Field(description="Output index")
    item: Dict[str, Any] = Field(description="Output item")


class ResponseFunctionCallArgumentsDeltaEvent(ResponseEventBase):
    type: Literal["response.function_call_arguments.delta"] = "response.function_call_arguments.delta"
    item_id: str = Field(description="Item ID")
    output_index: int = Field(description="Output index")
    call_id: str = Field(description="Call ID")
    delta: str = Field(description="Delta content")


class ResponseFunctionCallArgumentsDoneEvent(ResponseEventBase):
    type: Literal["response.function_call_arguments.done"] = "response.function_call_arguments.done"
    item_id: str = Field(description="Item ID")
    output_index: int = Field(description="Output index")
    call_id: str = Field(description="Call ID")
    name: str = Field(description="Function name")
    arguments: str = Field(description="Complete arguments")


class ResponseCompletedEvent(ResponseEventBase):
    type: Literal["response.completed"] = "response.completed"
    response: Dict[str, Any] = Field(description="Final response")
