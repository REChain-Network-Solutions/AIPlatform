<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

return new class extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('feedback_custom_forms', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('custom_fields_id')->nullable();
            $table->foreign('custom_fields_id')->references('id')->on('custom_fields')->onDelete('cascade')->onUpdate('cascade');
            $table->string('field_display_name');
            $table->string('field_name');
            $table->string('field_type')->default('text');
            $table->integer('field_order');
            $table->enum('status', ['active', 'inactive'])->default('active');
            $table->tinyInteger('required')->default('0');
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('feedback_custom_forms');
    }
};
