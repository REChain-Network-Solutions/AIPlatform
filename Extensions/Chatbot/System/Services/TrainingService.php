<?php

namespace App\Extensions\Chatbot\System\Services;

use App\Extensions\Chatbot\System\Models\Chatbot;

class TrainingService
{
    public Chatbot $chatbot;

    public function setChatbot(Chatbot $chatbot): static
    {
        $this->chatbot = $chatbot;

        return $this;
    }
}
