<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('einvoice_ai_drafts')) {
            Schema::create('einvoice_ai_drafts', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->unsignedBigInteger('user_id')->nullable();
                $table->unsignedBigInteger('client_id')->nullable();
                $table->json('payload');
                $table->timestamps();
                $table->index(['client_id', 'created_at']);
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('einvoice_ai_drafts');
    }
};
