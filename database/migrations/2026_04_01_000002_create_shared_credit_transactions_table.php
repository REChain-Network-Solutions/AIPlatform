<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('shared_credit_transactions', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->onDelete('cascade');
            $table->string('entity_key')->nullable();
            $table->string('engine_key')->nullable();
            $table->string('feature_type')->nullable();
            $table->string('action_type');
            $table->decimal('amount', 12, 4);
            $table->decimal('balance_after', 12, 4);
            $table->decimal('unit_cost', 10, 4)->default(0);
            $table->decimal('quantity', 10, 4)->default(0);
            $table->string('quality')->nullable();
            $table->json('metadata')->nullable();
            $table->string('description')->nullable();
            $table->timestamps();

            $table->index(['user_id', 'created_at']);
            $table->index(['user_id', 'action_type']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('shared_credit_transactions');
    }
};
