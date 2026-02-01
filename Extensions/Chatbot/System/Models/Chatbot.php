<?php

namespace App\Extensions\Chatbot\System\Models;

use App\Extensions\Chatbot\System\Enums\ColorModeEnum;
use App\Extensions\Chatbot\System\Enums\InteractionType;
use App\Extensions\Chatbot\System\Enums\PositionEnum;
use App\Models\User;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\BelongsTo;
use Illuminate\Database\Eloquent\Relations\HasMany;

class Chatbot extends Model
{
    protected $table = 'ext_chatbots';

    protected $fillable = [
        'uuid',
        'user_id',
        'interaction_type',
        'title',
        'bubble_message',
        'welcome_message',
        'connect_message',
        'instructions',
        'do_not_go_beyond_instructions',
        'language',
        'ai_model',
        'ai_embedding_model',
        'limit_per_minute',
        'show_pre_defined_questions',
        'pre_defined_questions',
        // customization start
        'logo',
        'avatar',
        'trigger_avatar_size',
        'trigger_background',
        'trigger_foreground',
        'color_mode',
        'color',
        'show_logo',
        'show_date_and_time',
        'show_average_response_time',
        'position',
        // customization end
        'active',
        'footer_link',
        'is_demo',
    ];

    protected $casts = [
        'color_mode'                    => ColorModeEnum::class,
        'position'                      => PositionEnum::class,
        'interaction_type'              => InteractionType::class,
        'do_not_go_beyond_instructions' => 'boolean',
        'limit_per_minute'              => 'integer',
        'show_pre_defined_questions'    => 'boolean',
        'pre_defined_questions'         => 'array',
        'active'                        => 'boolean',
        'user_id'                       => 'integer',
        'is_demo'                       => 'boolean',
    ];

    public function conversations(): HasMany
    {
        return $this->hasMany(ChatbotConversation::class, 'chatbot_id', 'id');
    }

    public function embeddings(): HasMany
    {
        return $this->hasMany(ChatbotEmbedding::class, 'chatbot_id', 'id');
    }

    public function user(): BelongsTo
    {
        return $this->belongsTo(User::class);
    }

    public function isActive(): bool
    {
        return (bool) $this->active;
    }
}
