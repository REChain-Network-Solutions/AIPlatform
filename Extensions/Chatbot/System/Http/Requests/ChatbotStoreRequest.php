<?php

namespace App\Extensions\Chatbot\System\Http\Requests;

use App\Domains\Entity\Enums\EntityEnum;
use App\Helpers\Classes\Helper;
use App\Models\Setting;
use Illuminate\Foundation\Http\FormRequest;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Str;

class ChatbotStoreRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'uuid'                          => ['required', 'string'],
            'user_id'                       => ['required', 'integer', 'exists:users,id'],
            'title'                         => ['required', 'string'],
            'bubble_message'                => ['required', 'string'],
            'welcome_message'               => ['required', 'string'],
            'interaction_type'              => ['required', 'string'],
            'instructions'                  => Helper::appIsNotDemo() ? ['required', 'string'] : ['sometimes', 'nullable', 'string'],
            'do_not_go_beyond_instructions' => ['required', 'boolean'],
            'language'                      => ['sometimes', 'nullable', 'string'],
            'ai_model'                      => ['required', 'string'],
            'ai_embedding_model'            => ['required', 'string'],
        ];
    }

    protected function prepareForValidation(): void
    {
        $this->merge([
            'uuid'               => Str::uuid()->toString(),
            'user_id'            => Auth::id(),
            'ai_model'			        => Setting::getCache()->openai_default_model,
            'ai_embedding_model' => $this->get('ai_embedding_model') ?: EntityEnum::TEXT_EMBEDDING_3_SMALL->value,
        ]);
    }
}
