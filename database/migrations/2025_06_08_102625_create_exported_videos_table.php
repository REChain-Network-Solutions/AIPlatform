<?php

use App\Enums\AiInfluencer\VideoStatusEnum;
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('exported_videos', function (Blueprint $table) {
            $table->id();
            $table->string('task_id')->unique()->index();
            $table->string('status')->default(VideoStatusEnum::IN_PROGRESS->value);
            $table->text('video_url')->nullable();
            $table->string('title')->nullable();
            $table->string('used_ai_tool')->default('topview');
            $table->text('cover_url')->nullable();
            $table->integer('video_duration')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('exported_videos');
    }
};
