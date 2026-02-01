<?php

namespace App\Extensions\Chatbot\System\Http\Requests\Train;

use App\Extensions\Chatbot\System\Models\Chatbot;
use App\Extensions\Chatbot\System\Models\ChatbotEmbedding;
use Illuminate\Foundation\Http\FormRequest;

class EmbedingRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'id'     => 'required|exists:' . (new Chatbot)->getTable() . ',id',
            'data'   => 'required|array',
            'data.*' => 'required|exists:' . (new ChatbotEmbedding)->getTable() . ',id',
        ];
    }
}
