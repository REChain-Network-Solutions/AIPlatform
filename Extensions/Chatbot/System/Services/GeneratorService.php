<?php

namespace App\Extensions\Chatbot\System\Services;

use App\Domains\Engine\Enums\EngineEnum;
use App\Domains\Entity\Enums\EntityEnum;
use App\Domains\Entity\Facades\Entity;
use App\Extensions\Chatbot\System\Generators\AnthropicGenerator;
use App\Extensions\Chatbot\System\Generators\Contracts\Generator;
use App\Extensions\Chatbot\System\Generators\GeminiGenerator;
use App\Extensions\Chatbot\System\Generators\OpenAIGenerator;
use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotConversation;
use App\Models\Setting;

class GeneratorService
{
    public string $prompt;

    public Chatbot $chatbot;

    public ChatbotConversation $conversation;

    public EntityEnum $entityEnum;

    public function generate(): string
    {
        $generator = $this->generator();

        $driver = Entity::driver($this->entityEnum)
            ->forUser($this->chatbot->user);

        if (! $driver->hasCreditBalance()) {
            return trans('You have no credits left. Please consider upgrading your plan.');
        }

        $generated = $generator
            ->setConversation($this->conversation)
            ->setChatbot($this->chatbot)
            ->setEntity($this->entityEnum)
            ->setPrompt($this->prompt)
            ->generate();

        $driver
            ->input($generated)
            ->calculateCredit()
            ->decreaseCredit();

        return $generated;
    }

    public function generator(): Generator
    {
        $setting = Setting::getCache();

        // TODO: chatbot model default openai_default_model
        //		$model = $this->chatbot->getAttribute('ai_model');
        $model = $setting->openai_default_model;

        $this->entityEnum = EntityEnum::fromSlug($setting->openai_default_model);

        $engine = $this->entityEnum->engine();

        return match ($engine) {
            EngineEnum::GEMINI    => app(GeminiGenerator::class),
            EngineEnum::ANTHROPIC => app(AnthropicGenerator::class),
            default               => app(OpenAIGenerator::class),
        };
    }

    public function getPrompt(): string
    {
        return $this->prompt;
    }

    public function setPrompt(string $prompt): static
    {
        $this->prompt = $prompt;

        return $this;
    }

    public function getChatbot(): Chatbot
    {
        return $this->chatbot;
    }

    public function setChatbot(Chatbot $chatbot): static
    {
        $this->chatbot = $chatbot;

        return $this;
    }

    public function getConversation(): ChatbotConversation
    {
        return $this->conversation;
    }

    public function setConversation(ChatbotConversation $conversation): static
    {
        $this->conversation = $conversation;

        return $this;
    }
}
