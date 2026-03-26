<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('csat_responses', function (Blueprint $table) {
            $table->id();
            $table->foreignId('survey_id')->constrained('csat_surveys')->cascadeOnDelete();
            $table->unsignedTinyInteger('rating'); // 1..5 (default scale)
            $table->unsignedBigInteger('user_id')->nullable();
            $table->string('email')->nullable();
            $table->text('comment')->nullable();
            $table->timestamps();
            $table->index(['survey_id','rating']);
        });
    }
    public function down(): void { Schema::dropIfExists('csat_responses'); }
};