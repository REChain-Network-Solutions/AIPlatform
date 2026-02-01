<?php

namespace App\Extensions\Chatbot\System\Services\OpenAI;

use App\Extensions\Chatbot\System\Models\Chatbot;

class MessageService
{
    /**
     * Chatbot instance
     */
    public Chatbot $chatbot;

    public function generateMessage() {}

    public function getChatbot(): Chatbot
    {
        return $this->chatbot;
    }

    public function setChatbot(Chatbot $chatbot): self
    {
        $this->chatbot = $chatbot;

        return $this;
    }
}
