<?php

namespace App\Models;

use App\Extensions\Canvas\System\Http\Models\UserTiptapContent;
use Exception;
use Illuminate\Database\Eloquent\Factories\HasFactory;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Database\Eloquent\Relations\MorphOne;
use Illuminate\Support\Facades\Schema;

class UserOpenaiChatMessage extends Model
{
    use HasFactory;

    protected $table = 'user_openai_chat_messages';

    protected $fillable = [
        'user_openai_chat_id',
        'user_id',
        'input',
        'response',
        'output',
        'hash',
        'credits',
        'words',
        'images',
        'pdfName',
        'pdfPath',
        'outputImage',
        'realtime',
        'is_chatbot',
    ];

    public function __construct(array $attributes = [])
    {
        parent::__construct($attributes);

        if (Schema::hasColumn($this->getTable(), 'model_slug')) {
            $this->fillable[] = 'model_slug';
        }

        if (Schema::hasColumn($this->getTable(), 'shared_uuid')) {
            $this->fillable[] = 'shared_uuid';
        }
    }

    protected static function booted(): void
    {
        static::created(static function ($message) {
            // Whenever a new message is added, mark chat as not empty
            if ($message->response !== 'First Initiation') {
                $message->chat()->update(['is_empty' => false]);
            }
        });

        static::deleted(static function ($message) {
            // flip back to empty if no messages remain
            if ($message->chat && $message->chat->messagesWithoutInitial()->count() === 0) {
                $message->chat->update(['is_empty' => true]);
            }
        });
    }

    public function chat()
    {
        return $this->belongsTo(UserOpenaiChat::class, 'user_openai_chat_id', 'id');
    }

    // tiptap edit result
    public function tiptapContent(): MorphOne
    {
        if (! class_exists(UserTiptapContent::class) || ! $this->tableExists('user_tiptap_contents')) {
            return $this->morphOne(self::class, 'user_openai_chat', 'user_id', 'id')->whereRaw('1 = 0');
        }

        return $this->morphOne(UserTiptapContent::class, 'save_contentable');
    }

    /**
     * Check if a table exists in the database
     */
    private function tableExists($tableName): bool
    {
        try {
            return Schema::hasTable($tableName);
        } catch (Exception $e) {
            return false;
        }
    }
}
