<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('nps_surveys', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->text('question')->default('How likely are you to recommend us to a friend or colleague?');
            $table->json('meta')->nullable();
            $table->timestamps();
        });
    }
    public function down(): void { Schema::dropIfExists('nps_surveys'); }
};