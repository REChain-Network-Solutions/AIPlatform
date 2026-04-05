<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (Schema::hasTable('module_install_logs')) {
            Schema::table('module_install_logs', function (Blueprint $table) {
                if (!Schema::hasColumn('module_install_logs', 'version')) {
                    $table->string('version')->nullable()->after('module_name');
                }
                if (!Schema::hasColumn('module_install_logs', 'status')) {
                    $table->string('status', 50)->default('pending')->after('version')->index();
                }
                if (!Schema::hasColumn('module_install_logs', 'manifest_data')) {
                    $table->json('manifest_data')->nullable()->after('status');
                }
                if (!Schema::hasColumn('module_install_logs', 'validation_results')) {
                    $table->json('validation_results')->nullable()->after('manifest_data');
                }
                if (!Schema::hasColumn('module_install_logs', 'validation_score')) {
                    $table->integer('validation_score')->default(0)->after('validation_results');
                }
                if (!Schema::hasColumn('module_install_logs', 'pre_install_snapshot')) {
                    $table->json('pre_install_snapshot')->nullable()->after('validation_score');
                }
                if (!Schema::hasColumn('module_install_logs', 'applied_repairs')) {
                    $table->json('applied_repairs')->nullable()->after('pre_install_snapshot');
                }
                if (!Schema::hasColumn('module_install_logs', 'repair_results')) {
                    $table->json('repair_results')->nullable()->after('applied_repairs');
                }
                if (!Schema::hasColumn('module_install_logs', 'module_source_hash')) {
                    $table->string('module_source_hash')->nullable()->after('repair_results');
                }
                if (!Schema::hasColumn('module_install_logs', 'module_source_size')) {
                    $table->unsignedBigInteger('module_source_size')->nullable()->after('module_source_hash');
                }
                if (!Schema::hasColumn('module_install_logs', 'can_rollback')) {
                    $table->boolean('can_rollback')->default(false)->after('module_source_size');
                }
                if (!Schema::hasColumn('module_install_logs', 'installed_at')) {
                    $table->timestamp('installed_at')->nullable()->after('can_rollback');
                }
                if (!Schema::hasColumn('module_install_logs', 'last_verified_at')) {
                    $table->timestamp('last_verified_at')->nullable()->after('installed_at');
                }
                if (!Schema::hasColumn('module_install_logs', 'installation_notes')) {
                    $table->text('installation_notes')->nullable()->after('last_verified_at');
                }
            });
        }

        if (Schema::hasTable('module_installation_logs') && !Schema::hasColumn('module_installation_logs', 'install_id')) {
            Schema::table('module_installation_logs', function (Blueprint $table) {
                $table->unsignedBigInteger('install_id')->nullable()->after('user_id')->index();
            });
        }
    }

    public function down(): void
    {
        if (Schema::hasTable('module_installation_logs') && Schema::hasColumn('module_installation_logs', 'install_id')) {
            Schema::table('module_installation_logs', function (Blueprint $table) {
                $table->dropColumn('install_id');
            });
        }
    }
};
