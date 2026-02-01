<?php

namespace App\Extensions\Chatbot\System\Http\Requests\Train;

use App\Extensions\Chatbot\System\Models\Chatbot;
use Illuminate\Foundation\Http\FormRequest;

class FileRequest extends FormRequest
{
    public function rules(): array
    {
        return [
            'id'     => 'required|exists:' . (new Chatbot)->getTable() . ',id',
            'file'   => 'required|file',
        ];
    }
}
