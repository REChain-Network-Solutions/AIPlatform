<?php

use App\Models\OpenAIGenerator;
use App\Models\OpenaiGeneratorChatCategory;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    public function up(): void
    {
        OpenAIGenerator::query()
            ->firstOrCreate([
                'slug' => 'ai_realtime_voice_chat',
            ], [
                'title'           => 'Realtime Voice Chat',
                'description'     => 'AI Realtime Voice Chat',
                'active'          => 1,
                'questions'       => '[{\"name\":\"your_description\",\"type\":\"textarea\",\"question\":\"Description\",\"select\":\"\"}]',
                'image'           => '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" stroke-width="2" stroke="black" fill="none" viewBox="0 0 24 24"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M6 4l6 16l6 -16" /></svg>',
                'premium'         => 0,
                'type'            => 'text',
                'prompt'          => null,
                'custom_template' => 0,
                'tone_of_voice'   => 0,
                'color'           => '#A3D6C2',
                'filters'         => 'blog',
            ]);

        OpenaiGeneratorChatCategory::query()
            ->firstOrCreate([
                'slug' => 'ai_realtime_voice_chat',
            ], [
                'name'             => 'Realtime Voice Chat',
                'short_name'       => 'RVC',
                'description'      => 'AI Realtime Voice Chat',
                'role'             => 'Voice Chatting Bot',
                'human_name'       => 'AI Realtime Voice Chat',
                'helps_with'       => 'I can assist you with voice chat',
                'prompt_prefix'    => 'As a Voice Chatting',
                'image'            => 'assets/img/chat-default.jpg',
                'color'            => '#EDBBBE',
                'chat_completions' => '[{"role": "system", "content": "You are a Voice Chatting assistant."}]',
                'plan'             => '',
                'category'         => '',
            ]);

    }

    public function down(): void {}
};
