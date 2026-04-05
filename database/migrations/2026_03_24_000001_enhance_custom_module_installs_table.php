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
        // Enhance custom_module_install_logs with state tracking
        Schema::table('custom_module_install_logs', function (Blueprint $table) {
            if (!Schema::hasColumn('custom_module_install_logs', 'module_name')) {
                $table->string('module_name')->after('file_name')->nullable()->index();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'version')) {
                $table->string('version')->after('module_name')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'status')) {
                $table->enum('status', ['pending', 'installed', 'partial', 'failed', 'rollback_needed'])
                    ->after('version')
                    ->default('pending')
                    ->index();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'manifest_data')) {
                $table->json('manifest_data')->after('status')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'validation_results')) {
                $table->json('validation_results')->after('manifest_data')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'validation_score')) {
                $table->integer('validation_score')->after('validation_results')->default(0);
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'pre_install_snapshot')) {
                $table->json('pre_install_snapshot')->after('validation_score')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'snapshot_path')) {
                $table->string('snapshot_path')->after('pre_install_snapshot')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'applied_repairs')) {
                $table->json('applied_repairs')->after('snapshot_path')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'repair_results')) {
                $table->json('repair_results')->after('applied_repairs')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'module_source_hash')) {
                $table->string('module_source_hash')->after('repair_results')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'module_source_size')) {
                $table->integer('module_source_size')->after('module_source_hash')->default(0);
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'can_rollback')) {
                $table->boolean('can_rollback')->after('module_source_size')->default(false);
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'installed_by')) {
                $table->string('installed_by')->after('can_rollback')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'installation_notes')) {
                $table->text('installation_notes')->after('installed_by')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'installed_at')) {
                $table->timestamp('installed_at')->after('installation_notes')->nullable();
            }
            if (!Schema::hasColumn('custom_module_install_logs', 'last_verified_at')) {
                $table->timestamp('last_verified_at')->after('installed_at')->nullable();
            }
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('custom_module_install_logs', function (Blueprint $table) {
            // Drop in reverse order
            $columns = [
                'last_verified_at', 'installed_at', 'installation_notes', 'installed_by',
                'can_rollback', 'module_source_size', 'module_source_hash', 'repair_results',
                'applied_repairs', 'snapshot_path', 'pre_install_snapshot', 'validation_score',
                'validation_results', 'manifest_data', 'status', 'version', 'module_name',
            ];
            
            foreach ($columns as $column) {
                if (Schema::hasColumn('custom_module_install_logs', $column)) {
                    $table->dropColumn($column);
                }
            }
        });
    }
};
