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
        Schema::create('projects', function (Blueprint $table) {
            $table->id();
            $table->string('name');
            $table->string('code', 50)->unique()->nullable();
            $table->text('description')->nullable();
            $table->string('status')->default('planning'); // Will be cast to enum
            $table->string('type')->default('client'); // Will be cast to enum
            $table->date('start_date')->nullable();
            $table->date('end_date')->nullable();
            $table->decimal('budget', 15, 2)->nullable();
            $table->string('priority')->default('medium'); // Will be cast to enum
            $table->string('color_code', 7)->default('#007bff');
            $table->boolean('is_billable')->default(true);
            $table->decimal('hourly_rate', 8, 2)->nullable();

            // Foreign keys (soft relationships)
            $table->unsignedBigInteger('client_id')->nullable(); // References companies table from CRMCore
            $table->unsignedBigInteger('project_manager_id')->nullable(); // References users table

            // Standard audit fields
            $table->unsignedBigInteger('created_by_id')->nullable();
            $table->unsignedBigInteger('updated_by_id')->nullable();

            $table->timestamps();
            $table->softDeletes();

            // Indexes for performance
            $table->index(['status', 'type']);
            $table->index(['client_id']);
            $table->index(['project_manager_id']);
            $table->index(['created_by_id']);
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('projects');
    }
};
