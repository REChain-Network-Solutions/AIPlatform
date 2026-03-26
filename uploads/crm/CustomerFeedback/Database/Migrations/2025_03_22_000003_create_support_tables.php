<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        // Feedback Channels
        Schema::create('feedback_channels', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('name');
            $table->string('slug')->unique();
            $table->text('description')->nullable();
            $table->string('icon')->nullable();
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id']);
        });

        // Feedback Types
        Schema::create('feedback_types', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('name');
            $table->string('slug');
            $table->text('description')->nullable();
            $table->enum('type_category', ['complaint', 'feedback', 'survey'])->default('feedback');
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id', 'type_category']);
        });

        // Feedback Groups
        Schema::create('feedback_groups', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('name');
            $table->text('description')->nullable();
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id']);
        });

        // Feedback Agent Groups (pivot)
        Schema::create('feedback_agent_groups', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->foreignId('group_id')->constrained('feedback_groups')->cascadeOnDelete();
            $table->foreignId('agent_id')->constrained('users')->cascadeOnDelete();
            $table->foreignId('added_by')->constrained('users')->cascadeOnDelete();
            $table->timestamps();
            $table->unique(['group_id', 'agent_id']);
            $table->index(['company_id']);
        });

        // Feedback Files
        Schema::create('feedback_files', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->foreignId('feedback_id')->constrained('feedback_tickets')->cascadeOnDelete();
            $table->foreignId('reply_id')->nullable()->constrained('feedback_replies')->cascadeOnDelete();
            
            $table->string('filename');
            $table->string('file_path');
            $table->integer('file_size');
            $table->string('mime_type');
            
            $table->timestamps();
            $table->index(['feedback_id']);
            $table->index(['reply_id']);
        });

        // Feedback Tags
        Schema::create('feedback_tags_list', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('name');
            $table->text('description')->nullable();
            $table->string('color')->nullable();
            $table->timestamps();
            $table->index(['company_id']);
        });

        // Feedback Tags Pivot
        Schema::create('feedback_tags', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->foreignId('feedback_id')->constrained('feedback_tickets')->cascadeOnDelete();
            $table->foreignId('tag_id')->constrained('feedback_tags_list')->cascadeOnDelete();
            $table->timestamps();
            $table->unique(['feedback_id', 'tag_id']);
            $table->index(['company_id']);
        });

        // Reply Templates
        Schema::create('feedback_reply_templates', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('name');
            $table->text('description')->nullable();
            $table->longText('message');
            $table->enum('reply_type', ['auto', 'manual'])->default('manual');
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id']);
        });

        // Custom Forms
        Schema::create('feedback_custom_forms', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('name');
            $table->text('description')->nullable();
            $table->json('fields');
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id']);
        });

        // Email Settings
        Schema::create('feedback_email_settings', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('imap_host');
            $table->integer('imap_port')->default(993);
            $table->enum('imap_encryption', ['ssl', 'tls', 'none'])->default('ssl');
            $table->string('imap_username');
            $table->text('imap_password');
            $table->string('email_address');
            $table->boolean('auto_reply')->default(false);
            $table->longText('reply_message')->nullable();
            $table->timestamp('last_sync')->nullable();
            $table->timestamps();
            $table->index(['company_id']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('feedback_email_settings');
        Schema::dropIfExists('feedback_custom_forms');
        Schema::dropIfExists('feedback_reply_templates');
        Schema::dropIfExists('feedback_tags');
        Schema::dropIfExists('feedback_tags_list');
        Schema::dropIfExists('feedback_files');
        Schema::dropIfExists('feedback_agent_groups');
        Schema::dropIfExists('feedback_groups');
        Schema::dropIfExists('feedback_types');
        Schema::dropIfExists('feedback_channels');
    }
};
