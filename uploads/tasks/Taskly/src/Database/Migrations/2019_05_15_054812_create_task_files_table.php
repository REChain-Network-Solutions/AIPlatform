<?php

use Illuminate\Support\Facades\Schema;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Database\Migrations\Migration;

class CreateTaskFilesTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        if(!Schema::hasTable('task_files'))
        {
            Schema::create('task_files', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->string('file');
                $table->string('name');
                $table->string('extension')->nullable();
                $table->string('file_size')->nullable();
                $table->integer('task_id');
                $table->string('user_type');
                $table->integer('created_by');
                $table->timestamps();
            });
        }

    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('task_files');
    }
}
