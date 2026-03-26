<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void {
        Schema::create('csat_surveys', function (Blueprint $table) {
            $table->id();
            $table->string('title');
            $table->text('question')->default('How satisfied are you with your experience?');
            $table->json('scale')->nullable(); // e.g. ["Very dissatisfied",...,"Very satisfied"]
            $table->json('meta')->nullable();
            $table->timestamps();
        });
    }
    public function down(): void { Schema::dropIfExists('csat_surveys'); }
};