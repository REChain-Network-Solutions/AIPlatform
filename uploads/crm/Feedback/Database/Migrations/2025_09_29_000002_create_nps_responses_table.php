<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('nps_responses', function (Blueprint $table) {
            $table->id();
            $table->foreignId('survey_id')->constrained('nps_surveys')->cascadeOnDelete();
            $table->unsignedTinyInteger('score'); // 0..10
            $table->unsignedBigInteger('user_id')->nullable();
            $table->string('email')->nullable();
            $table->text('comment')->nullable();
            $table->timestamps();
            $table->index(['survey_id','score']);
        });
    }
    public function down(): void { Schema::dropIfExists('nps_responses'); }
};