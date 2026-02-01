<?php

namespace App\Extensions\Chatbot\System\Http\Requests;

use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Str;

class ChatbotUpdateRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'title'                         => ['required', 'string'],
            'bubble_message'                => ['required', 'string'],
            'welcome_message'               => ['required', 'string'],
            'instructions'                  => ['required', 'string'],
            'do_not_go_beyond_instructions' => ['required', 'boolean'],
            'language'                      => ['sometimes', 'nullable', 'string'],
            'ai_model'                      => ['required', 'string'],
        ];
    }

    protected function prepareForValidation(): void
    {
        $this->merge([
            'uuid'                          => Str::uuid()->toString(),
            'user_id'                       => Auth::id(),
        ]);
    }
}
