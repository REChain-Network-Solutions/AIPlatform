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
        Schema::table(self::$prefix . '_chatbot_embeddings', function (Blueprint $table) {
            if (! Schema::hasColumn(self::$prefix . '_chatbot_embeddings', 'type')) {
                $table->string('type')->nullable()->default(EmbeddingTypeEnum::text->value)->after('embedding');
            }
        });
    }

    public function down(): void {}
};
