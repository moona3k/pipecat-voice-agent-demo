import argparse
import os
import datetime

from dotenv import load_dotenv
from loguru import logger

# Pipecat core components for building the voice agent pipeline
from pipecat.audio.vad.silero import SileroVADAnalyzer  # Voice Activity Detection - detects when someone is speaking
from pipecat.pipeline.pipeline import Pipeline  # Main pipeline that connects all services together
from pipecat.pipeline.runner import PipelineRunner  # Runs and manages the pipeline execution
from pipecat.pipeline.task import PipelineParams, PipelineTask  # Task management and configuration
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext  # Manages conversation history/context

# Pipecat services for different AI capabilities
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService  # Text-to-Speech service
from pipecat.services.deepgram.stt import DeepgramSTTService  # Speech-to-Text service
from pipecat.services.openai.llm import OpenAILLMService  # LLM for generating responses

# Pipecat transport layers for different communication methods
from pipecat.transports.base_transport import BaseTransport, TransportParams  # Base transport interface
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketParams  # For Twilio/WebSocket connections (optional)
from pipecat.transports.services.daily import DailyParams  # For Daily.co video/audio calls (optional)

load_dotenv(override=True)

# Transport configuration dictionary
transport_params = {
    # WebRTC - for web browser audio connections
    "webrtc": lambda: TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
}

async def init_voice_agent(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info("Starting Gaia - Mutual of Omaha Voice Agent")

    # Speech-to-Text service
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))

    # Text-to-Speech service
    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVENLABS_API_KEY"),
        voice_id="EXAVITQu4vr4xnSDxMaL",  # professional female voice
    )

    # LLM service
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Define persona & behavior for the voice agent
    messages = [
        {
            "role": "system",
            "content": """You are Gaia, a friendly and professional insurance assistant for Mutual of Omaha. Your personality is:

PERSONALITY TRAITS:
- Warm, caring, and genuinely interested in helping families
- Patient and understanding, especially with seniors
- Knowledgeable but never condescending
- Trustworthy and reliable - you've been helping families for years
- Empathetic listener who remembers personal details

YOUR ROLE:
1. Warmly introduce yourself as Gaia from Mutual of Omaha
2. Identify their insurance needs with genuine care
3. Qualify prospects for Medicare, Life, Disability, or Annuity products  
4. Capture information for specialist follow-up
5. Never be pushy - focus on helping them find the right protection

COMPANY CONTEXT:
- Mutual of Omaha has served families since 1909 - over 115 years of trust
- We specialize in Medicare Supplement, Life Insurance, and Financial products
- Our mission is protecting families' financial security and peace of mind

CONVERSATION STYLE:
- Speak naturally and conversationally (under 20 words per response)
- Use their name once you learn it
- Ask ONE question at a time
- Show genuine concern: "I want to make sure you're properly protected"
- Reference your experience: "In my years helping families..."

QUALIFICATION PROCESS:
1. "Hi, this is Gaia from Mutual of Omaha. I'm here to help with your insurance needs."
2. Get their name and ask if they have a few minutes to chat
3. "What brings you to Mutual of Omaha today?" (discover their interest)
4. Qualify: age range, current coverage, timeline, specific concerns
5. Collect contact info with care: "What's the best number to reach you?"
6. "I'll have one of our specialists call you. They'll take great care of you."

Start by introducing yourself as Gaia and ask how you can help them today.""",
        },
    ]

    # Create a context manager to maintain conversation history
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Build the main pipeline - this is the heart of the voice agent
    pipeline = Pipeline(
        [
            transport.input(),  # 1. Get audio input from the user (microphone)
            stt,  # 2. Convert speech to text using Deepgram
            context_aggregator.user(),  # 3. Add user's text to conversation history
            llm,  # 4. Generate AI response using OpenAI
            tts,  # 5. Convert AI's text response to speech using ElevenLabs
            transport.output(),  # 6. Send audio output to the user (speakers)
            context_aggregator.assistant(),  # 7. Add AI's response to conversation history
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Event handler: What happens when a user connects to the voice agent
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"🟢 CLIENT CONNECTED - Client ID: {client}")
        
        # Determine context based on time of day
        current_hour = datetime.datetime.now().hour
        
        if current_hour < 8 or current_hour > 17:  # Before 8 AM or after 5 PM
            context_message = "Explain that you're available after hours to help with their insurance needs."
        else:
            context_message = "Explain that you're here to help while our other agents are with other families."
            
        logger.info(f"💬 Context message: {context_message}")
        
        messages.append({
            "role": "system", 
            "content": f"Start by warmly introducing yourself as Gaia from Mutual of Omaha. {context_message} Ask how you can help them today."
        })

        # Trigger the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    # Event handler: What happens when a user disconnects
    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"🔴 CLIENT DISCONNECTED - Client ID: {client}")
        await task.cancel()

    # Create a pipeline runner to manage the execution
    runner = PipelineRunner(handle_sigint=handle_sigint)

    # Start running the pipeline task
    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    main(init_voice_agent, transport_params=transport_params)
