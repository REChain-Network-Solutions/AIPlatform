<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        // NPS Surveys
        Schema::create('nps_surveys', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('title');
            $table->text('description')->nullable();
            $table->text('question')->default('How likely are you to recommend us to a friend or colleague?');
            $table->json('meta')->nullable();
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id']);
        });

        // NPS Responses
        Schema::create('nps_responses', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->foreignId('nps_survey_id')->constrained('nps_surveys')->cascadeOnDelete();
            $table->foreignId('feedback_ticket_id')->nullable()->constrained('feedback_tickets')->nullOnDelete();
            $table->foreignId('user_id')->constrained('users')->cascadeOnDelete();
            
            $table->integer('score')->min(1)->max(10);
            $table->longText('feedback')->nullable();
            $table->json('metadata')->nullable();
            
            $table->timestamps();
            $table->index(['nps_survey_id']);
            $table->index(['user_id']);
            $table->index(['feedback_ticket_id']);
        });

        // CSAT Surveys
        Schema::create('csat_surveys', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->string('title');
            $table->text('description')->nullable();
            $table->text('question')->default('How satisfied are you with our service?');
            $table->integer('scale_min')->default(1);
            $table->integer('scale_max')->default(5);
            $table->json('meta')->nullable();
            $table->boolean('status')->default(true);
            $table->timestamps();
            $table->index(['company_id']);
        });

        // CSAT Responses
        Schema::create('csat_responses', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->foreignId('csat_survey_id')->constrained('csat_surveys')->cascadeOnDelete();
            $table->foreignId('feedback_ticket_id')->nullable()->constrained('feedback_tickets')->nullOnDelete();
            $table->foreignId('user_id')->constrained('users')->cascadeOnDelete();
            
            $table->integer('score');
            $table->longText('feedback')->nullable();
            $table->json('metadata')->nullable();
            
            $table->timestamps();
            $table->index(['csat_survey_id']);
            $table->index(['user_id']);
            $table->index(['feedback_ticket_id']);
        });

        // Feedback Insights
        Schema::create('feedback_insights', function (Blueprint $table) {
            $table->id();
            $table->foreignId('company_id')->constrained('companies')->cascadeOnDelete();
            $table->foreignId('feedback_ticket_id')->constrained('feedback_tickets')->cascadeOnDelete();
            
            $table->enum('insight_type', ['sentiment', 'category', 'priority', 'action', 'trend']);
            $table->string('title');
            $table->longText('description');
            $table->float('confidence_score')->default(0.0);
            $table->longText('suggested_action')->nullable();
            $table->json('tags')->nullable();
            $table->json('metadata')->nullable();
            
            $table->timestamps();
            $table->index(['feedback_ticket_id']);
            $table->index(['insight_type']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('feedback_insights');
        Schema::dropIfExists('csat_responses');
        Schema::dropIfExists('csat_surveys');
        Schema::dropIfExists('nps_responses');
        Schema::dropIfExists('nps_surveys');
    }
};
