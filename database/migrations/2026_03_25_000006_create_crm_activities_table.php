<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('crm_activities', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->enum('type', ['note', 'call', 'email', 'meeting', 'task', 'deadline'])->default('note');
            $table->string('subject');
            $table->text('body')->nullable();
            $table->timestamp('due_at')->nullable();
            $table->timestamp('done_at')->nullable();
            $table->boolean('is_done')->default(false);

            // Polymorphic-style optional links
            $table->foreignId('crm_contact_id')->nullable()->constrained('crm_contacts')->nullOnDelete();
            $table->foreignId('crm_company_id')->nullable()->constrained('crm_companies')->nullOnDelete();
            $table->foreignId('crm_deal_id')->nullable()->constrained('crm_deals')->nullOnDelete();

            $table->timestamps();
            $table->softDeletes();

            $table->index('user_id');
            $table->index('type');
            $table->index('is_done');
            $table->index('due_at');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('crm_activities');
    }
};
