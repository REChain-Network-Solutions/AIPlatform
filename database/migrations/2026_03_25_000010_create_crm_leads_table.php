<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('crm_leads', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('crm_lead_status_id')->nullable()->constrained('crm_lead_statuses')->nullOnDelete();
            $table->foreignId('crm_lead_source_id')->nullable()->constrained('crm_lead_sources')->nullOnDelete();

            // Lead contact info (may not yet be a full contact)
            $table->string('title');                     // lead title / subject
            $table->string('contact_name')->nullable();
            $table->string('contact_email')->nullable();
            $table->string('contact_phone')->nullable();
            $table->string('company_name')->nullable();
            $table->string('website')->nullable();

            // Qualification
            $table->decimal('value', 15, 2)->default(0);
            $table->string('currency', 3)->default('USD');
            $table->text('description')->nullable();
            $table->json('custom_fields')->nullable();

            // Kanban ordering within status column
            $table->integer('order')->default(0);

            // Assignment
            $table->foreignId('assigned_to_user_id')->nullable()->constrained('users')->nullOnDelete();

            // Conversion
            $table->foreignId('converted_to_contact_id')->nullable()->constrained('crm_contacts')->nullOnDelete();
            $table->foreignId('converted_to_deal_id')->nullable()->constrained('crm_deals')->nullOnDelete();
            $table->timestamp('converted_at')->nullable();

            $table->timestamps();
            $table->softDeletes();

            $table->index('user_id');
            $table->index('crm_lead_status_id');
            $table->index('crm_lead_source_id');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('crm_leads');
    }
};
