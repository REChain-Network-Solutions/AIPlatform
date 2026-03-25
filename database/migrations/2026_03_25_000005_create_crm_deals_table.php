<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('crm_deals', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('pipeline_id')->constrained('crm_pipelines')->cascadeOnDelete();
            $table->foreignId('stage_id')->constrained('crm_pipeline_stages')->cascadeOnDelete();
            $table->foreignId('crm_contact_id')->nullable()->constrained('crm_contacts')->nullOnDelete();
            $table->foreignId('crm_company_id')->nullable()->constrained('crm_companies')->nullOnDelete();
            $table->string('title');
            $table->decimal('value', 15, 2)->default(0);
            $table->string('currency', 3)->default('USD');
            $table->integer('probability')->nullable(); // override stage probability
            $table->date('expected_close_date')->nullable();
            $table->date('closed_at')->nullable();
            $table->enum('status', ['open', 'won', 'lost'])->default('open');
            $table->text('description')->nullable();
            $table->text('lost_reason')->nullable();
            $table->integer('order')->default(0); // position within stage (kanban)
            $table->json('custom_fields')->nullable();
            $table->timestamps();
            $table->softDeletes();

            $table->index('user_id');
            $table->index('pipeline_id');
            $table->index('stage_id');
            $table->index('status');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('crm_deals');
    }
};
