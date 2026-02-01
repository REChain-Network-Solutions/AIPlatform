<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('pebblely')) {
            return;
        }

        Schema::create('pebblely', function (Blueprint $table) {
            $table->id();
            $table->integer('user_id');
            $table->string('image');
            $table->timestamps();
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('pebblely');
    }
};
