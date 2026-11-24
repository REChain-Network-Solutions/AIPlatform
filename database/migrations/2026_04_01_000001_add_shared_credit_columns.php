<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('users', function (Blueprint $table) {
            if (! Schema::hasColumn('users', 'credit_system_type')) {
                $table->string('credit_system_type')->default('separated')->after('entity_credits');
            }
            if (! Schema::hasColumn('users', 'shared_credits')) {
                $table->decimal('shared_credits', 12, 4)->default(0)->after('credit_system_type');
            }
            if (! Schema::hasColumn('users', 'shared_credits_expires_at')) {
                $table->timestamp('shared_credits_expires_at')->nullable()->after('shared_credits');
            }
        });

        Schema::table('plans', function (Blueprint $table) {
            if (! Schema::hasColumn('plans', 'credit_system_type')) {
                $table->string('credit_system_type')->default('separated')->after('type');
            }
            if (! Schema::hasColumn('plans', 'shared_credits_amount')) {
                $table->decimal('shared_credits_amount', 12, 4)->default(0)->after('credit_system_type');
            }
            if (! Schema::hasColumn('plans', 'shared_credit_model_overrides')) {
                $table->json('shared_credit_model_overrides')->nullable()->after('shared_credits_amount');
            }
            if (! Schema::hasColumn('plans', 'shared_credit_feature_limits')) {
                $table->json('shared_credit_feature_limits')->nullable()->after('shared_credit_model_overrides');
            }
        });

        Schema::table('teams', function (Blueprint $table) {
            if (! Schema::hasColumn('teams', 'credit_system_type')) {
                $table->string('credit_system_type')->default('separated')->after('entity_credits');
            }
            if (! Schema::hasColumn('teams', 'shared_credits')) {
                $table->decimal('shared_credits', 12, 4)->default(0)->after('credit_system_type');
            }
        });
    }

    public function down(): void
    {
        Schema::table('users', function (Blueprint $table) {
            $table->dropColumn(['credit_system_type', 'shared_credits', 'shared_credits_expires_at']);
        });

        Schema::table('plans', function (Blueprint $table) {
            $table->dropColumn(['credit_system_type', 'shared_credits_amount', 'shared_credit_model_overrides', 'shared_credit_feature_limits']);
        });

        Schema::table('teams', function (Blueprint $table) {
            $table->dropColumn(['credit_system_type', 'shared_credits']);
        });
    }
};
