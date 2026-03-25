<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('crm_contacts', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete(); // owner (workspace)
            $table->string('first_name');
            $table->string('last_name')->nullable();
            $table->string('email')->nullable();
            $table->string('phone')->nullable();
            $table->string('job_title')->nullable();
            $table->string('company_name')->nullable();           // freeform; links via crm_company_id
            $table->unsignedBigInteger('crm_company_id')->nullable(); // FK added after crm_companies created
            $table->string('linkedin_url')->nullable();
            $table->string('twitter_handle')->nullable();
            $table->string('website')->nullable();
            $table->text('address')->nullable();
            $table->string('city')->nullable();
            $table->string('state')->nullable();
            $table->string('country', 2)->nullable();
            $table->string('avatar_path')->nullable();
            $table->enum('status', ['lead', 'prospect', 'customer', 'churned', 'partner'])->default('lead');
            $table->text('notes')->nullable();
            $table->json('custom_fields')->nullable();
            $table->timestamp('last_contacted_at')->nullable();
            $table->timestamps();
            $table->softDeletes();

            $table->index('user_id');
            $table->index('email');
            $table->index('status');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('crm_contacts');
    }
};
