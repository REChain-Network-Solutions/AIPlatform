<?php

use App\Extensions\Chatbot\System\Enums\EmbeddingTypeEnum;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public static $prefix = 'ext';

    public function up(): void
    {
        if (Schema::hasTable('ext_chatbot_embeddings')) {
            return;
        }
        Schema::create(self::$prefix . '_chatbot_embeddings', function (Blueprint $table) {
            $table->id();
            $table->foreignId('chatbot_id')->constrained(self::$prefix . '_chatbots')->cascadeOnDelete();
            $table->string('engine');
            $table->string('title')->nullable();
            $table->string('file')->nullable();
            $table->string('url')->nullable();
            $table->longText('content')->nullable();
            $table->json('embedding')->nullable();
            $table->string('type')->nullable()->default(EmbeddingTypeEnum::text->value);
            $table->timestamp('trained_at')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists(self::$prefix . '_chatbot_embeddings');
    }
};
