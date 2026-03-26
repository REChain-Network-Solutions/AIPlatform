<?php
if (!function_exists('getLogData')) {
	function getLogData($primaryDataKey = "logType")
	{
		$secondaryDataKey = ($primaryDataKey == "logType") ? "logDate" : "logType";

		$folderPath = get_instance()->config->item('log_path');
		$folderPath = !empty($folderPath) ? $folderPath : APPPATH . '/logs';

		$extension = get_instance()->config->item('log_file_extension');
		$extension = !empty($extension) ? $extension : 'php';

		$files = directory_map($folderPath);
		$files = array_reverse($files);

		$logData = ['count' => 0, 'data' => []];

		foreach ($files as $file) {
			$filePath = $folderPath . "/" . $file;
			$ext = pathinfo($filePath, PATHINFO_EXTENSION);

			if (is_file($filePath) && $ext === $extension) {
				$content = file_get_contents($filePath);

				preg_match_all('/^(.*?) - (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) --> (.*)$/m', $content, $matches, PREG_SET_ORDER);
				$matches = array_reverse($matches);

				foreach ($matches as $match) {
					[$fullMatch, $logType, $logTime, $logMsg] = $match;
					$logDate = substr($logTime, 0, 10); // Extract date from log_time

					// Increment counts
					$logData['count']++;
					$logData['data'][${$primaryDataKey}]['count'] = ($logData['data'][${$primaryDataKey}]['count'] ?? 0) + 1;
					$logData['data'][${$primaryDataKey}]['data'][${$secondaryDataKey}]['count'] = ($logData['data'][${$primaryDataKey}]['data'][${$secondaryDataKey}]['count'] ?? 0) + 1;

					// Store data
					$logData['data'][${$primaryDataKey}]['data'][${$secondaryDataKey}]['data'][] = (object) ["message" => $logMsg, "time" => $logTime, 'level' => $logType];
				}
			}
		}

		return $logData;
	}
}

if (!function_exists('getColorByCategory')) {
	function getColorByCategory($category)
	{
		$color = !empty(get_option($category . '_color')) ? get_option($category . '_color') : '#64748b';

		return $color;
	}
}

if (!function_exists('displayEnvironmentMessage')) {
	function displayEnvironmentMessage()
	{
		$message = new app\services\messages\DevelopmentEnvironment();

		$errorHtml = '';
		if (ENVIRONMENT == 'development') {
			$errorHtml = '<div class="alert alert-warning">';
			$errorHtml .= $message->getMessage();
			$errorHtml .= '</div>';
		}

		return $errorHtml;
	}
}
