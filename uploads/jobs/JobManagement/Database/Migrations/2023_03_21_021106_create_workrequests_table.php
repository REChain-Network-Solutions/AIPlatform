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
        Schema::create('workrequests', function (Blueprint $table) {
            $table->increments('id');
            $table->unsignedInteger('company_id')->nullable();
            $table->foreign('company_id')->references('id')->on('companies')->onDelete('cascade')->onUpdate('cascade');
            $table->unsignedInteger('complaint_id')->nullable();
            $table->foreign('complaint_id')->references('id')->on('complaint')->onDelete('cascade')->onUpdate('cascade');
            $table->string('wr_no');
            $table->unsignedInteger('assign_to')->nullable();
            $table->foreign('assign_to')->references('id')->on('users')->onDelete('cascade')->onUpdate('cascade');
            $table->dateTime('check_time')->nullable();
            $table->mediumText('remark')->nullable();
            $table->boolean('charge_by_tenant')->default(0);
            $table->boolean('status_wo')->default(0);
            $table->enum('status', ['uncheck', 'checked'])->default('uncheck');
            $table->unsignedInteger('house_id')->nullable();
            $table->foreign('house_id')->references('id')->on('houses')->onDelete('cascade')->onUpdate('cascade');
            $table->mediumText('problem')->nullable();
            $table->string('image')->nullable();
            $table->unsignedInteger('created_by');
            $table->foreign('created_by')->references('id')->on('users')->onDelete('cascade')->onUpdate('cascade');
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
        Schema::dropIfExists('workrequests');
    }
};
