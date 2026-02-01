<?php

namespace App\Extensions\Chatbot\System\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;

class ChatbotHistoryStoreRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'prompt' => 'required|string',
        ];
    }
}
