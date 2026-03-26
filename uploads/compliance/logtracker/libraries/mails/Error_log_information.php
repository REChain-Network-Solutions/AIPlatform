<?php

defined('BASEPATH') or exit('No direct script access allowed');

class Error_log_information extends App_mail_template
{
	protected $email;

	protected $errorLevel;

	protected $errorTime;

	protected $errorMessage;

	public $slug = 'error-log-information';

	public function __construct($email, $errorLevel, $errorTime, $errorMessage)
	{
		parent::__construct();

		$this->email = $email;
		$this->errorLevel = $errorLevel;
		$this->errorTime = $errorTime;
		$this->errorMessage = $errorMessage;
	}

	public function build()
	{
		$this->to($this->email)->set_merge_fields('logtracker_merge_fields', $this->errorLevel, $this->errorTime, $this->errorMessage);
	}
}
