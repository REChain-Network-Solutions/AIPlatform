<?php

namespace App\Extensions\Chatbot\System\Models;

use App\Extensions\Chatbot\System\Enums\EmbeddingTypeEnum;
use Illuminate\Database\Eloquent\Model;

class ChatbotEmbedding extends Model
{
    protected $table = 'ext_chatbot_embeddings';

    protected $fillable = [
        'chatbot_id',
        'engine',
        'title',
        'file',
        'url',
        'content',
        'embedding',
        'type',
        'trained_at',
    ];

    protected $casts = [
        'embedding' => 'json',
        'type'      => EmbeddingTypeEnum::class,
    ];
}
