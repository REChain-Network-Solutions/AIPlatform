<?php

namespace App\Extensions\Chatbot\System\Http\Requests\Train;

use App\Extensions\Chatbot\System\Enums\EmbeddingTypeEnum;
use App\Extensions\Chatbot\System\Models\Chatbot;
use Illuminate\Foundation\Http\FormRequest;

class DataRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'id'         => 'required|exists:' . (new Chatbot)->getTable() . ',id',
            'type'       => ['sometimes', 'nullable', 'in:' . implode(',', EmbeddingTypeEnum::toArray())],
        ];
    }
}
