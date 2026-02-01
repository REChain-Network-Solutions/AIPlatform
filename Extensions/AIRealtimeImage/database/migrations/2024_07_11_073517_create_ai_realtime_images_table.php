<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('ai_realtime_images', function (Blueprint $table) {
            $table->id();
            $table->bigInteger('user_id');
            $table->string('prompt')->nullable();
            $table->string('image')->nullable();
            $table->string('disk')->default('public')->nullable();
            $table->json('payload')->nullable();
            $table->json('response')->nullable();
            $table->string('status')->nullable();
            $table->string('model')->nullable();
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('ai_realtime_images');
    }
};
