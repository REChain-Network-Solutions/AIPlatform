<?php

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
        Schema::create('project_members', function (Blueprint $table) {
            $table->id();
            $table->unsignedBigInteger('project_id');
            $table->unsignedBigInteger('user_id');
            $table->string('role')->default('member'); // Will be cast to enum
            $table->decimal('hourly_rate', 8, 2)->nullable();
            $table->integer('allocation_percentage')->default(100);
            $table->timestamp('joined_at')->nullable();
            $table->timestamp('left_at')->nullable();

            // Standard audit fields
            $table->unsignedBigInteger('created_by_id')->nullable();

            $table->timestamps();

            // Composite unique index to prevent duplicate memberships
            $table->unique(['project_id', 'user_id'], 'project_user_unique');

            // Indexes for performance
            $table->index(['project_id']);
            $table->index(['user_id']);
            $table->index(['role']);
            $table->index(['created_by_id']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('project_members');
    }
};
