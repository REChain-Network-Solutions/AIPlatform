<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
    public function up(): void
    {
        if (Schema::hasTable('inspection_template_items')) {
            return;
        }

        Schema::create('inspection_template_items', function (Blueprint $table) {
            $table->bigIncrements('id');

            $table->unsignedBigInteger('template_id');
            $table->string('item_name', 191);
            $table->string('standard', 191)->nullable();
            $table->unsignedInteger('sort_order')->default(0);
            $table->boolean('is_required')->default(false);

            $table->timestamps();

            // Short FK name (MySQL 64-char safe)
            $table->foreign('template_id', 'fk_insp_tpl_items_tpl_id')
                ->references('id')
                ->on('inspection_templates')
                ->onDelete('cascade')
                ->onUpdate('cascade');

            $table->index(['template_id', 'sort_order']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('inspection_template_items');
    }
};
