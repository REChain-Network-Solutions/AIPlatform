<?php

namespace App\Livewire;

use App\Extensions\AiVideoPro\System\Models\UserFall;
use App\Helpers\Classes\MarketplaceHelper;
use App\Models\ExportedVideo;
use App\Models\UserOpenai;
use Exception;
use Illuminate\Support\Collection;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Storage;
use Illuminate\Support\Str;
use Livewire\Component;
use Livewire\WithFileUploads;

class MediaManagerModal extends Component
{
    use WithFileUploads;

    public string $activeFilter = 'Images';

    public bool $isProcessingFiles = false;

    public string $processingMessage = '';

    public bool $showModal = false;

    public string $searchTerm = '';

    public array $allowedTypes = ['all']; // 'image', 'video', 'file', or 'all'

    /** @var int[] */
    public array $selectedOtherFiles = [];

    public int $loadedOtherFilesCount = 0;

    public bool $hasMoreOtherFiles = true;

    /** @var int[] */
    public array $selectedImages = [];

    public array $selectedStockImages = [];

    public array $selectedStockVideos = [];

    /** @var int[] */
    public array $selectedVideos = [];

    public $uploadingFiles = [];

    /** @var string[] */
    public array $filters = [];

    public array $sortButtons = [];

    public string $sort = 'created_at';

    public string $sortAscDesc = 'desc';

    // Infinite scroll properties
    public int $loadedImagesCount = 0;

    public int $loadedVideosCount = 0;

    public int $loadPerBatch = 30; // Load 30 items at a time

    public bool $hasMoreImages = true;

    public bool $hasMoreVideos = true;

    public bool $isLoading = false;

    // Upload properties
    public bool $isUploading = false;

    public string $uploadProgress = '';

    public array $uploadErrors = [];

    public array $uploadedFiles = [];

    public bool $allowMultipleSelection = false;

    protected $listeners = [
        'openMediaManager' => 'openModal',
        'loadMore'         => 'loadMore',
    ];

    protected function rules()
    {
        $maxSize = (int) setting('media_max_size', 25) * 1024; // Convert MB to KB for Laravel validation

        return [
            'uploadingFiles.*' => "file|max:{$maxSize}",
        ];
    }

    protected function messages()
    {
        $maxSize = setting('media_max_size', 25);

        return [
            'uploadingFiles.*.max'  => __("Each file must be smaller than {$maxSize}MB."),
            'uploadingFiles.*.file' => __('The uploaded item must be a valid file.'),
        ];
    }

    public function mount(): void
    {
        $this->filters = array_filter([
            'Upload Files',
            'Images',
            'Videos',
            'Other Files',
            empty(setting('pexels_api_key')) ? null : 'Stock Images',
            empty(setting('pexels_api_key')) ? null : 'Stock Videos',
            MarketplaceHelper::isRegistered('google-drive') ? 'Google Drive' : null,
        ]);

        $this->sortButtons = [
            ['label' => __('Date'), 'sort' => 'created_at'],
            ['label' => __('Title'), 'sort' => 'title'],
            ['label' => __('Input'), 'sort' => 'input'],
        ];
    }

    public function updatedSearchTerm(): void
    {
        // Reset counters when search term changes
        $this->resetLoadingCounters();
        // Load initial batch immediately after search
        $this->loadInitialBatch();
        // Emit event to reinitialize intersection observer
        $this->dispatch('searchUpdated');
    }

    public function updatedActiveFilter(): void
    {
        // Reset counters when filter changes
        $this->resetLoadingCounters();
        // Load initial batch for new filter
        $this->loadInitialBatch();
        // Emit event to reinitialize intersection observer
        $this->dispatch('searchUpdated');
    }

    public function updatedSort(): void
    {
        // Reset counters when sort changes
        $this->resetLoadingCounters();
        // Reload data with new sort
        $this->loadInitialBatch();
    }

    public function updatedSortAscDesc(): void
    {
        // Reset counters when sort direction changes
        $this->resetLoadingCounters();
        // Reload data with new sort direction
        $this->loadInitialBatch();
    }

    public function updatedUploadingFiles(): void
    {
        $isAdmin = auth()->user()?->isAdmin() ?? false;
        $maxFiles = $isAdmin ? PHP_INT_MAX : (int) setting('media_max_files', 5);
        $maxSizeMB = $isAdmin ? PHP_INT_MAX : (int) setting('media_max_size', 25);
        $allowedTypes = $isAdmin ? [] : explode(',', setting('media_allowed_types', 'jpg,png,gif,webp,svg,mp4,avi,mov,wmv,flv,webm,mp3,wav,m4a,pdf,doc,docx,xls,xlsx'));

        if (! $isAdmin) {
            $allowedTypes = array_map('trim', $allowedTypes);
            $allowedTypes = array_map('strtolower', $allowedTypes);
        }

        // Set processing state immediately
        $this->isProcessingFiles = true;
        $this->processingMessage = $isAdmin ? __('Admin mode: Processing files...') : __('Validating selected files...');
        $this->uploadErrors = [];

        // Emit event to update UI immediately
        $this->dispatch('fileProcessingStarted');

        if (empty($this->uploadingFiles)) {
            $this->isProcessingFiles = false;
            $this->processingMessage = '';

            return;
        }

        try {
            // Update processing message for large file validation
            if (count($this->uploadingFiles) > 1) {
                $this->processingMessage = $isAdmin
                    ? __('Admin mode: Processing ' . count($this->uploadingFiles) . ' files...')
                    : __('Validating ' . count($this->uploadingFiles) . ' files...');
            }

            // Validate file count using dynamic setting (skip for admin)
            if (! $isAdmin && count($this->uploadingFiles) > $maxFiles) {
                $this->uploadErrors[] = __("Maximum {$maxFiles} files allowed per upload.");
                $this->uploadingFiles = array_slice($this->uploadingFiles, 0, $maxFiles);
            }

            $validFiles = [];
            $processedCount = 0;

            // Validate each file with progress updates (skip validation for admin)
            foreach ($this->uploadingFiles as $index => $file) {
                $processedCount++;

                if (! $file) {
                    continue;
                }

                // Update progress for multiple files
                if (count($this->uploadingFiles) > 1) {
                    $this->processingMessage = $isAdmin
                        ? __("Admin mode: Processing file {$processedCount} of " . count($this->uploadingFiles) . '...')
                        : __("Validating file {$processedCount} of " . count($this->uploadingFiles) . '...');
                }

                $fileName = $file->getClientOriginalName();
                $fileSize = $file->getSize();

                // Skip all validation for admin users
                if ($isAdmin) {
                    $validFiles[] = $file;

                    continue;
                }

                // Check file size using dynamic setting (only for non-admin)
                $maxSizeBytes = $maxSizeMB * 1024 * 1024;
                if ($fileSize > $maxSizeBytes) {
                    $this->uploadErrors[] = __("File '{$fileName}' is larger than {$maxSizeMB}MB.");

                    continue;
                }

                // Check file type using dynamic setting (only for non-admin)
                $extension = strtolower($file->guessExtension());
                if (! in_array($extension, $allowedTypes)) {
                    $this->uploadErrors[] = __("File '{$fileName}' has an unsupported format. Allowed types: " . implode(', ', $allowedTypes));

                    continue;
                }

                // Additional validation for very large files (show progress)
                if ($fileSize > 10 * 1024 * 1024) { // Files larger than 10MB
                    $this->processingMessage = __("Processing large file: {$fileName}...");
                    // Add a small delay to allow UI to update for very large files
                    usleep(100000); // 0.1 second
                }

                $validFiles[] = $file;
            }

            // Update the files array with only valid files
            $this->uploadingFiles = $validFiles;

            // Final processing message
            if (count($validFiles) > 0) {
                $this->processingMessage = $isAdmin
                    ? __('Admin mode: Files ready for upload!')
                    : __('Files ready for upload!');
            }

        } catch (Exception $e) {
            $this->uploadErrors[] = 'Error processing files: ' . $e->getMessage();
        } finally {
            // Clear processing state
            $this->isProcessingFiles = false;
            $this->processingMessage = '';

            // Emit completion event
            $this->dispatch('fileProcessingCompleted', [
                'validFileCount' => count($this->uploadingFiles),
                'hasErrors'      => ! empty($this->uploadErrors),
            ]);

            // Automatically start upload if there are valid files
            if (! empty($this->uploadingFiles)) {
                $this->uploadFiles(); // Trigger upload automatically
            }
        }
    }

    public function clearProcessingState(): void
    {
        $this->isProcessingFiles = false;
        $this->processingMessage = '';
    }

    public function uploadFiles(): void
    {
        $this->validateOnly('uploadingFiles');

        if (empty($this->uploadingFiles)) {
            $this->uploadErrors[] = __('No files selected for upload.');

            return;
        }

        $this->isUploading = true;
        $this->uploadProgress = __('Starting upload...');
        $this->uploadedFiles = [];
        $this->uploadErrors = [];

        try {
            foreach ($this->uploadingFiles as $index => $file) {
                if (! $file) {
                    continue;
                }

                $this->uploadProgress = __('Uploading file ' . ($index + 1) . ' of ' . count($this->uploadingFiles) . '...');

                if (! isFileSecure($file)) {
                    $this->uploadErrors[] = __("File '{$file->getClientOriginalName()}' is not allowed for security reasons.");

                    continue;
                }

                $extension = strtolower($file->guessExtension());
                $isImage = in_array($extension, ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg']);
                $isVideo = in_array($extension, ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm']);
                $fileType = $isImage ? 'images' : ($isVideo ? 'videos' : 'other');

                // Create user-specific folder
                $userFolder = 'media/' . $fileType . '/u-' . auth()->id() . '/';

                // Ensure directory exists
                if (! Storage::disk('public')->exists($userFolder)) {
                    Storage::disk('public')->makeDirectory($userFolder);
                }

                // Generate unique filename to prevent conflicts
                $originalName = pathinfo($file->getClientOriginalName(), PATHINFO_FILENAME);
                $sanitizedName = Str::slug($originalName);
                $uniqueName = $sanitizedName . '_' . time() . '_' . Str::random(5) . '.' . $extension;

                // Store the file
                $filePath = $file->storeAs($userFolder, $uniqueName, 'public');

                if ($filePath) {

                    $fullPath = Storage::disk('public')->path($filePath);
                    if (! validateUploadedFile($fullPath, $extension)) {
                        Storage::disk('public')->delete($filePath);
                        $this->uploadErrors[] = __("File '{$file->getClientOriginalName()}' failed security validation.");

                        continue;
                    }

                    $this->uploadedFiles[] = [
                        'name' => $file->getClientOriginalName(),
                        'path' => $filePath,
                        'type' => $fileType,
                        'size' => $file->getSize(),
                    ];
                } else {
                    $this->uploadErrors[] = __("Failed to upload '{$file->getClientOriginalName()}'.");
                }
            }

            $this->uploadProgress = __('Upload completed!');

            // Clear the upload files after successful upload
            $this->uploadingFiles = [];

            // Show success message
            if (count($this->uploadedFiles) > 0) {
                $this->dispatch('uploadComplete', [
                    'message' => count($this->uploadedFiles) . ' file(s) uploaded successfully!',
                    'files'   => $this->uploadedFiles,
                ]);

                // FIXED: Switch to appropriate tab based on the first uploaded file type
                if (count($this->uploadedFiles) > 0) {
                    $firstFileType = $this->uploadedFiles[0]['type'];

                    // Switch to the correct tab based on file type
                    switch ($firstFileType) {
                        case 'images':
                            $this->activeFilter = 'Images';

                            break;
                        case 'videos':
                            $this->activeFilter = 'Videos';

                            break;
                        case 'other':
                            $this->activeFilter = 'Other Files';

                            break;
                        default:
                            // Stay on Upload Files tab if type is unknown
                            break;
                    }

                    $this->resetLoadingCounters();
                    $this->loadInitialBatch();
                }
            }

        } catch (Exception $e) {
            $this->uploadErrors[] = 'Upload failed: ' . $e->getMessage();
        } finally {
            $this->isUploading = false;

            $this->dispatch('clearUploadProgress');
        }
    }

    private function resetLoadingCounters(): void
    {
        $this->loadedImagesCount = 0;
        $this->loadedVideosCount = 0;
        $this->loadedOtherFilesCount = 0;
        $this->hasMoreImages = true;
        $this->hasMoreVideos = true;
        $this->hasMoreOtherFiles = true;
    }

    private function loadInitialBatch(): void
    {
        if ($this->activeFilter === 'Images') {
            $this->loadedImagesCount = $this->loadPerBatch;
        } elseif ($this->activeFilter === 'Videos') {
            $this->loadedVideosCount = $this->loadPerBatch - 25;
        } elseif ($this->activeFilter === 'Other Files') {
            $this->loadedOtherFilesCount = $this->loadPerBatch;
        }
    }

    public function loadMore(): void
    {
        if ($this->isLoading) {
            return;
        }

        $this->isLoading = true;

        if ($this->activeFilter === 'Images' && $this->hasMoreImages) {
            $this->loadedImagesCount += $this->loadPerBatch;
        } elseif ($this->activeFilter === 'Videos' && $this->hasMoreVideos) {
            $this->loadedVideosCount += $this->loadPerBatch;
        } elseif ($this->activeFilter === 'Other Files' && $this->hasMoreOtherFiles) {
            $this->loadedOtherFilesCount += $this->loadPerBatch;
        }

        $this->isLoading = false;
    }

    public function openModal(array $allowedTypes = ['all'], bool $isMultiple = false): void
    {
        $this->allowedTypes = $allowedTypes;
        $this->showModal = true;
        $this->allowMultipleSelection = $isMultiple;
        // Load initial batch when opening modal
        $this->loadInitialBatch();
    }

    public function closeModal(): void
    {
        $this->showModal = false;
        $this->clearFiles();
        $this->uploadErrors = [];
        $this->uploadProgress = '';
        $this->uploadedFiles = [];
        $this->isProcessingFiles = false;
        $this->processingMessage = '';
    }

    // Updated getStockVideos method with better video file selection
    protected function getStockVideos(bool $getAllForSelection = false): Collection
    {
        $searchTerm = trim($this->searchTerm);
        if (empty($searchTerm)) {
            return collect();
        }

        $perPage = (int) setting('pexels_video_count', 6);
        $url = "https://api.pexels.com/videos/search?query={$searchTerm}&per_page={$perPage}";

        try {
            $response = Http::withHeaders([
                'Authorization' => setting('pexels_api_key'),
            ])->get($url);

            if ($response->successful()) {
                $data = $response->json();

                return collect($data['videos'] ?? [])->map(function ($video) {
                    // Get video files and sort by quality preference
                    $videoFiles = collect($video['video_files'] ?? []);

                    // First try to get a smaller/compressed version for preview
                    $previewFile = $videoFiles->first(function ($file) {
                        return in_array($file['quality'], ['sd', 'hd']) &&
                            isset($file['width']) && $file['width'] <= 640;
                    });

                    // If no small preview found, get the smallest available
                    if (! $previewFile) {
                        $previewFile = $videoFiles->sortBy('width')->first();
                    }

                    // For download, prefer HD quality
                    $downloadFile = $videoFiles->first(function ($file) {
                        return $file['quality'] === 'hd';
                    }) ?? $videoFiles->first(function ($file) {
                        return $file['quality'] === 'sd';
                    }) ?? $videoFiles->first();

                    return (object) [
                        'id'          => 'pexels_video_' . $video['id'],
                        'title'       => $video['tags'] ? ucfirst($video['tags']) : "Video by {$video['user']['name']}",
                        'url'         => $downloadFile['link'] ?? '', // Main video URL for download
                        'preview_url' => $previewFile['link'] ?? '', // Smaller preview URL for in-browser playback
                        'thumbnail'   => $video['image'] ?? '',
                        'duration'    => $video['duration'] ?? 0,
                        'source'      => 'pexels_video',
                        'created_at'  => now(),
                        'format_date' => now()->format('M d, Y'),
                        'width'       => $downloadFile['width'] ?? 0,
                        'height'      => $downloadFile['height'] ?? 0,
                        'file_type'   => $downloadFile['file_type'] ?? 'video/mp4',
                        'quality'     => $downloadFile['quality'] ?? 'sd',
                        // Additional debug info
                        'preview_width'  => $previewFile['width'] ?? 0,
                        'preview_height' => $previewFile['height'] ?? 0,
                    ];
                });
            }
        } catch (Exception $e) {
            Log::error('Pexels Video API Error: ' . $e->getMessage());
        }

        return collect();
    }

    public function downloadAndInsertStockVideos(): void
    {
        if (empty($this->selectedStockVideos)) {
            return;
        }

        $stockVideos = $this->getStockVideos(true);
        $selectedStockVideos = $stockVideos->whereIn('id', $this->selectedStockVideos);

        if ($selectedStockVideos->isEmpty()) {
            return;
        }

        $downloadedVideos = collect();
        $errors = [];

        foreach ($selectedStockVideos as $stockVideo) {
            try {
                // Extract the actual video ID from the prefixed ID
                $videoId = str_replace('pexels_video_', '', $stockVideo->id);

                // Download the video
                $response = Http::timeout(60)->get($stockVideo->url); // Increased timeout for videos

                if ($response->successful()) {
                    // Generate filename
                    $fileName = 'stock_video_' . $videoId . '_' . time() . '.mp4';
                    $userFolder = 'media/videos/u-' . auth()->id() . '/';

                    // Ensure directory exists
                    if (! Storage::disk('public')->exists($userFolder)) {
                        Storage::disk('public')->makeDirectory($userFolder);
                    }

                    $filePath = $userFolder . $fileName;
                    Storage::disk('public')->put($filePath, $response->body());

                    // Create downloaded video object with consistent structure
                    $downloadedVideo = (object) [
                        'id'         => 'downloaded_video_' . $videoId . '_' . time(),
                        'title'      => $stockVideo->title,
                        'input'      => $stockVideo->title,
                        'output_url' => url('/uploads/' . $filePath),
                        'url'        => url('/uploads/' . $filePath),
                        'created_at' => now(),
                        'extension'  => 'mp4',
                        'type'       => 'video',
                        'source'     => 'stock_video_downloaded',
                        'file_path'  => $filePath,
                    ];

                    $downloadedVideos->push($downloadedVideo);
                } else {
                    $errors[] = 'Failed to download: ' . $stockVideo->title;
                }
            } catch (Exception $e) {
                Log::error('Stock video download failed: ' . $e->getMessage());
                $errors[] = 'Download error for: ' . $stockVideo->title;
            }
        }

        // Show errors if any
        if (! empty($errors)) {
            foreach ($errors as $error) {
                $this->dispatch('stockVideoDownloadFailed', [
                    'message' => $error,
                ]);
            }
        }

        // If we have successfully downloaded videos, insert them
        if ($downloadedVideos->isNotEmpty()) {
            // Format the downloaded videos for frontend consumption
            $formattedItems = $downloadedVideos->map(function ($item) {
                return [
                    'id'         => $item->id,
                    'url'        => $item->output_url,
                    'title'      => $item->title,
                    'input'      => $item->input,
                    'extension'  => $item->extension,
                    'type'       => 'video',
                    'created_at' => $item->created_at,
                ];
            });

            // Dispatch success message
            $this->dispatch('stockVideosDownloaded', [
                'message' => count($downloadedVideos) . ' stock video(s) downloaded and inserted successfully!',
                'files'   => $formattedItems->toArray(),
            ]);

            // Dispatch media selected event with downloaded videos
            $this->dispatch('mediaSelected', [
                'type'  => 'video',
                'items' => $formattedItems->toArray(),
            ]);

            // Reset selections and close modal
            $this->selectedStockVideos = [];
            $this->closeModal();

            // Switch to Videos tab to show the downloaded videos
            $this->activeFilter = 'Videos';
            $this->resetLoadingCounters();
            $this->loadInitialBatch();
        }
    }

    public function toggleSelect(string $type, string $id): void
    {
        $selectedProperty = match ($type) {
            'image'      => 'selectedImages',
            'video'      => 'selectedVideos',
            'other'      => 'selectedOtherFiles',
            'stockImage' => 'selectedStockImages',
            'stockVideo' => 'selectedStockVideos',
            default      => 'selectedImages'
        };

        $id = (string) $id;
        if (in_array($id, $this->$selectedProperty, true)) {
            $this->$selectedProperty = collect($this->$selectedProperty)
                ->reject(fn ($selectedId) => $selectedId === $id)
                ->values()
                ->toArray();
        } else {
            $this->{$selectedProperty}[] = $id;
        }
    }

    public function changeFilter(string $filter): void
    {
        if ($this->isLoading) {
            return;
        }

        $this->isLoading = true;

        try {
            $this->activeFilter = $filter;
            $this->selectedImages = [];
            $this->selectedVideos = [];
            $this->selectedOtherFiles = [];
            $this->selectedStockImages = [];
            $this->selectedStockVideos = [];

            // Clear upload data when switching away from upload tab
            if ($filter !== 'Upload Files') {
                $this->clearFiles();
                $this->uploadErrors = [];
                $this->uploadProgress = '';
            }

            // Reset loading counters
            $this->resetLoadingCounters();

            // Load initial batch for the new filter
            $this->loadInitialBatch();

            // Emit event to update UI
            $this->dispatch('filterChanged', ['filter' => $filter]);
        } finally {
            $this->isLoading = false;
        }
    }

    /**
     * Download and insert selected stock images
     */
    public function downloadAndInsertStockImages(): void
    {
        if (empty($this->selectedStockImages)) {
            return;
        }

        $stockImages = $this->getStockImages(true);
        $selectedStockImages = $stockImages->whereIn('id', $this->selectedStockImages);

        if ($selectedStockImages->isEmpty()) {
            return;
        }

        $downloadedImages = collect();
        $errors = [];

        foreach ($selectedStockImages as $stockImage) {
            try {
                // Extract the actual image ID from the prefixed ID
                $imageId = str_replace('pexels_', '', $stockImage->id);

                // Download the image
                $response = Http::timeout(30)->get($stockImage->url);

                if ($response->successful()) {
                    // Generate filename
                    $fileName = 'stock_' . $imageId . '_' . time() . '.jpg';
                    $userFolder = 'media/images/u-' . auth()->id() . '/';

                    // Ensure directory exists
                    if (! Storage::disk('public')->exists($userFolder)) {
                        Storage::disk('public')->makeDirectory($userFolder);
                    }

                    $filePath = $userFolder . $fileName;
                    Storage::disk('public')->put($filePath, $response->body());

                    // Create downloaded image object with consistent structure
                    $downloadedImage = (object) [
                        'id'         => 'downloaded_' . $imageId . '_' . time(),
                        'title'      => $stockImage->title,
                        'input'      => $stockImage->title,
                        'output_url' => url('/uploads/' . $filePath),
                        'url'        => url('/uploads/' . $filePath),
                        'created_at' => now(),
                        'extension'  => 'jpg',
                        'type'       => 'image',
                        'source'     => 'stock_downloaded',
                        'file_path'  => $filePath,
                    ];

                    $downloadedImages->push($downloadedImage);
                } else {
                    $errors[] = 'Failed to download: ' . $stockImage->title;
                }
            } catch (Exception $e) {
                Log::error('Stock image download failed: ' . $e->getMessage());
                $errors[] = 'Download error for: ' . $stockImage->title;
            }
        }

        // Show errors if any
        if (! empty($errors)) {
            foreach ($errors as $error) {
                $this->dispatch('stockImageDownloadFailed', [
                    'message' => $error,
                ]);
            }
        }

        // If we have successfully downloaded images, insert them
        if ($downloadedImages->isNotEmpty()) {
            // Format the downloaded images for frontend consumption
            $formattedItems = $downloadedImages->map(function ($item) {
                return [
                    'id'         => $item->id,
                    'url'        => $item->output_url,
                    'title'      => $item->title,
                    'input'      => $item->input,
                    'extension'  => $item->extension,
                    'type'       => 'image',
                    'created_at' => $item->created_at,
                ];
            });

            // Dispatch success message
            $this->dispatch('stockImagesDownloaded', [
                'message' => count($downloadedImages) . ' stock image(s) downloaded and inserted successfully!',
                'files'   => $formattedItems->toArray(),
            ]);

            // Dispatch media selected event with downloaded images
            $this->dispatch('mediaSelected', [
                'type'  => 'image',
                'items' => $formattedItems->toArray(),
            ]);

            // Reset selections and close modal
            $this->selectedStockImages = [];
            $this->closeModal();

            // Switch to Images tab to show the downloaded images
            $this->activeFilter = 'Images';
            $this->resetLoadingCounters();
            $this->loadInitialBatch();
        }
    }

    public function changeSort(string $sort): void
    {
        if ($this->sort === $sort) {
            $this->sortAscDesc = $this->sortAscDesc === 'asc' ? 'desc' : 'asc';
        } else {
            $this->sort = $sort;
            $this->sortAscDesc = 'desc';
        }
    }

    protected function getUserUploadedOtherFiles(): Collection
    {
        if (! auth()->check()) {
            return collect();
        }

        $userUploadedMediaFolder = 'media/other/u-' . auth()->id() . '/';

        // Create the folder if it doesn't exist
        if (! Storage::disk('public')->exists($userUploadedMediaFolder)) {
            Storage::disk('public')->makeDirectory($userUploadedMediaFolder);
        }
        $files = Storage::disk('public')->files($userUploadedMediaFolder);

        return collect($files)
            ->map(function ($file) {
                $filename = basename($file);
                $extension = strtolower(pathinfo($file, PATHINFO_EXTENSION));

                try {
                    $lastModified = Storage::disk('public')->lastModified($file);
                    $createdAt = \Carbon\Carbon::createFromTimestamp($lastModified);
                } catch (Exception $e) {
                    $createdAt = \Carbon\Carbon::now();
                }

                $fileUrl = url('/uploads/' . $file);

                return (object) [
                    'id'          => 'file_' . md5($file),
                    'title'       => pathinfo($filename, PATHINFO_FILENAME),
                    'filename'    => $filename,
                    'extension'   => $extension,
                    'input'       => '',
                    'output_url'  => $fileUrl,
                    'url'         => $fileUrl,
                    'created_at'  => $createdAt,
                    'format_date' => $createdAt->format('M d, Y'),
                    'source'      => 'uploaded',
                    'file_path'   => $file,
                    'file_size'   => Storage::disk('public')->size($file),
                ];
            })
            ->values();
    }

    /**
     * Fetch other files with limit and offset for infinite scroll
     */
    protected function getOtherFiles(bool $getAllForSelection = false): Collection
    {
        // Get uploaded other files
        $uploadedOtherFiles = $this->getUserUploadedOtherFiles();

        // Filter by search term if provided
        $searchTerm = trim($this->searchTerm);
        if (! empty($searchTerm)) {
            $uploadedOtherFiles = $uploadedOtherFiles->filter(function ($item) use ($searchTerm) {
                return Str::contains(strtolower($item->title), strtolower($searchTerm)) ||
                    Str::contains(strtolower($item->filename), strtolower($searchTerm));
            });
        }

        // Apply sorting
        $allOtherFiles = $this->sortCollection($uploadedOtherFiles);

        // Handle pagination if not getting all for selection
        if (! $getAllForSelection) {
            $totalCount = $allOtherFiles->count();
            $this->hasMoreOtherFiles = $this->loadedOtherFilesCount < $totalCount;

            return $allOtherFiles->take($this->loadedOtherFilesCount);
        }

        return $allOtherFiles;
    }

    public function clearFiles(): void
    {
        $this->uploadingFiles = [];
        $this->uploadErrors = [];
        $this->uploadedFiles = [];
        $this->isProcessingFiles = false;
        $this->processingMessage = '';
    }

    public function clearSearch(): void
    {
        $this->searchTerm = '';
        $this->resetLoadingCounters();
        $this->loadInitialBatch();

        // Emit event to reinitialize intersection observer
        $this->dispatch('searchUpdated');
    }

    public function insertSelected(string $type): void
    {
        $selectedItems = match ($type) {
            'image'      => $this->selectedImages,
            'video'      => $this->selectedVideos,
            'other'      => $this->selectedOtherFiles,
            'stockImage' => $this->selectedStockImages,
            'stockVideo' => $this->selectedStockVideos,
            default      => []
        };

        if (empty($selectedItems)) {
            return;
        }

        // If single selection is enforced, only pick the first item
        if (! $this->allowMultipleSelection) {
            $selectedItems = [reset($selectedItems)];
        }

        // Get the actual media items based on selected IDs
        $mediaItems = match ($type) {
            'image'      => $this->getImages(true),
            'video'      => $this->getVideos(true),
            'other'      => $this->getOtherFiles(true),
            'stockImage' => $this->getStockImages(true),
            'stockVideo' => $this->getStockVideos(true),
            default      => collect()
        };
        $selectedMediaItems = $mediaItems->whereIn('id', $selectedItems);

        // Format the data for frontend consumption
        $formattedItems = $selectedMediaItems->map(function ($item) use ($type) {
            return [
                'id'         => $item->id,
                'url'        => $item->output_url ?? $item->url,
                'title'      => $item->title ?? $item->input ?? $item->filename,
                'input'      => $item->input ?? '',
                'extension'  => $item->extension ?? '',
                'type'       => $type,
                'created_at' => $item->created_at,
            ];
        });

        // Dispatch an event with the selected items
        $this->dispatch('mediaSelected', [
            'type'  => $type,
            'items' => $formattedItems->toArray(),
        ]);

        // Reset selections and close modal
        match ($type) {
            'image'      => $this->selectedImages = [],
            'video'      => $this->selectedVideos = [],
            'other'      => $this->selectedOtherFiles = [],
            'stockImage' => $this->selectedStockImages = [],
            'stockVideo' => $this->selectedStockVideos = [],
            default      => $this->selectedImages = [],
        };

        $this->closeModal();
    }

    /**
     * Get uploaded files from user's media folder
     */
    protected function getUserUploadedFiles(string $type = 'image'): Collection
    {
        if (! auth()->check()) {
            return collect();
        }

        $userUploadedMediaFolder = 'media/' . ($type === 'image' ? 'images' : 'videos') . '/u-' . auth()->id() . '/';

        // Create the folder if it doesn't exist
        if (! Storage::disk('public')->exists($userUploadedMediaFolder)) {
            Storage::disk('public')->makeDirectory($userUploadedMediaFolder);
        }

        // Get allowed extensions from settings
        $allowedTypesFromSettings = explode(',', setting('media_allowed_types', 'jpg,png,gif,webp,svg,mp4,avi,mov,wmv,flv,webm,mp3,wav,m4a,pdf,doc,docx,xls,xlsx'));
        $allowedTypesFromSettings = array_map('trim', $allowedTypesFromSettings);

        // Filter by type (image or video)
        if ($type === 'image') {
            $imageExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'];
            $allowedExtensions = array_intersect($allowedTypesFromSettings, $imageExtensions);
        } else {
            $videoExtensions = ['mp4', 'avi', 'mov', 'wmv', 'flv', 'webm'];
            $allowedExtensions = array_intersect($allowedTypesFromSettings, $videoExtensions);
        }

        $files = Storage::disk('public')->files($userUploadedMediaFolder);

        return collect($files)
            ->filter(function ($file) use ($allowedExtensions) {
                $extension = strtolower(pathinfo($file, PATHINFO_EXTENSION));

                return in_array($extension, $allowedExtensions);
            })
            ->map(function ($file) {
                $filename = basename($file);

                try {
                    $lastModified = Storage::disk('public')->lastModified($file);
                    $createdAt = \Carbon\Carbon::createFromTimestamp($lastModified);
                } catch (Exception $e) {
                    // Fallback to current time if we can't get the file timestamp
                    $createdAt = \Carbon\Carbon::now();
                }

                $fileUrl = url('/uploads/' . $file);

                return (object) [
                    'id'          => 'file_' . md5($file),
                    'title'       => pathinfo($filename, PATHINFO_FILENAME),
                    'filename'    => $filename,
                    'input'       => '',
                    'output_url'  => $fileUrl,
                    'url'         => $fileUrl,
                    'created_at'  => $createdAt,
                    'format_date' => $createdAt->format('M d, Y'),
                    'source'      => 'uploaded',
                    'file_path'   => $file,
                    'file_size'   => Storage::disk('public')->size($file),
                ];
            })
            ->values();
    }

    /**
     * Fetch images with limit and offset for infinite scroll - Enhanced to include uploaded files
     */
    protected function getImages(bool $getAllForSelection = false): Collection
    {
        // Get database images
        $query = UserOpenai::query()
            ->where('user_id', auth()->id())
            ->whereNotNull('output')
            ->where('output', '!=', '')
            ->whereHas('generator', function ($q) {
                $q->where('type', 'image');
            })
            ->with(['generator' => function ($q) {
                $q->select('id', 'type', 'title');
            }]);

        // Apply search filter at database level
        $searchTerm = trim($this->searchTerm);
        if (! empty($searchTerm)) {
            $query->where(function ($q) use ($searchTerm) {
                $q->where('title', 'like', "%{$searchTerm}%")
                    ->orWhere('input', 'like', "%{$searchTerm}%");
            });
        }

        $dbImages = $query->get()->map(function ($item) {
            $item->source = 'database';

            return $item;
        });

        // Get uploaded files
        $uploadedImages = $this->getUserUploadedFiles('image');

        // Filter uploaded files by search term if provided
        if (! empty($searchTerm)) {
            $uploadedImages = $uploadedImages->filter(function ($item) use ($searchTerm) {
                return Str::contains(strtolower($item->title), strtolower($searchTerm)) ||
                    Str::contains(strtolower($item->filename), strtolower($searchTerm));
            });
        }

        // Merge both collections
        $allImages = $dbImages->concat($uploadedImages);

        // Apply sorting
        $allImages = $this->sortCollection($allImages);

        // Handle pagination if not getting all for selection
        if (! $getAllForSelection) {
            $totalCount = $allImages->count();
            $this->hasMoreImages = $this->loadedImagesCount < $totalCount;

            return $allImages->take($this->loadedImagesCount);
        }

        return $allImages;
    }

    /**
     * Fetch stock images with limit and offset for infinite scroll
     */
    protected function getStockImages(bool $getAllForSelection = false): Collection
    {
        $searchTerm = trim($this->searchTerm);
        if (empty($searchTerm)) {
            return collect();
        }

        $perPage = (int) setting('pexels_image_count', 20);
        $url = "https://api.pexels.com/v1/search?query={$searchTerm}&per_page={$perPage}";

        try {
            $response = Http::withHeaders([
                'Authorization' => setting('pexels_api_key'),
            ])->get($url);

            if ($response->successful()) {
                $data = $response->json();

                return collect($data['photos'] ?? [])->map(function ($photo) {
                    return (object) [
                        'id'          => 'pexels_' . $photo['id'],
                        'title'       => $photo['alt'] ?: "Photo by {$photo['photographer']}",
                        'url'         => $photo['src']['original'],
                        'thumbnail'   => $photo['src']['medium'],
                        'source'      => 'pexels',
                        'created_at'  => now(),
                        'format_date' => now()->format('M d, Y'),
                    ];
                });
            }
        } catch (Exception $e) {
            Log::error('Pexels API Error: ' . $e->getMessage());
        }

        return collect();
    }

    /**
     * Fetch videos with limit and offset for infinite scroll - Enhanced to include uploaded files
     */
    protected function getVideos(bool $getAllForSelection = false): Collection
    {
        // Early return if not on videos tab and not getting all for selection
        if (! $getAllForSelection && $this->activeFilter !== 'Videos') {
            return collect();
        }

        $searchTerm = trim($this->searchTerm);
        $allVideos = collect();

        try {
            // Get database videos
            $query = UserOpenai::query()
                ->where('user_id', auth()->id())
                ->where('status', 'COMPLETED')
                ->whereNotNull('output')
                ->where('output', '!=', '')
                ->whereHas('generator', function ($q) {
                    $q->where('type', 'video');
                })
                ->with(['generator' => function ($q) {
                    $q->select('id', 'type', 'title');
                }]);

            // Apply search filter at database level for UserOpenai
            if (! empty($searchTerm)) {
                $query->where(function ($q) use ($searchTerm) {
                    $q->where('title', 'like', "%{$searchTerm}%")
                        ->orWhere('input', 'like', "%{$searchTerm}%");
                });
            }

            $dbVideos = $query->get()->map(function ($item) {
                $item->source = 'database';

                return $item;
            });

            $allVideos = $allVideos->concat($dbVideos);

            // Get UserFall videos if the class exists (with error handling)
            if (class_exists('App\Extensions\AiVideoPro\System\Models\UserFall')) {
                try {
                    $userFallQuery = UserFall::query()
                        ->where('user_id', auth()->id())
                        ->where('status', 'complete')
                        ->whereNotNull('video_url')
                        ->where('video_url', '!=', '');

                    // Apply search filter for UserFall videos
                    if (! empty($searchTerm)) {
                        $userFallQuery->where('prompt', 'like', "%{$searchTerm}%");
                    }

                    $userFallVideos = $userFallQuery->get()->map(function ($item) {
                        return (object) [
                            'id'          => 'userfall_' . $item->id,
                            'title'       => $this->generateTitleFromPrompt($item->prompt),
                            'input'       => $item->prompt,
                            'output'      => $item->video_url,
                            'output_url'  => $item->video_url,
                            'url'         => $item->video_url,
                            'created_at'  => $item->created_at,
                            'format_date' => $item->created_at->format('M d, Y'),
                            'source'      => 'userfall',
                            'model'       => $item->model ?? 'veo2',
                            'request_id'  => $item->request_id,
                            'is_demo'     => $item->is_demo ?? 0,
                            'generator'   => (object) [
                                'id'    => null,
                                'type'  => 'video',
                                'title' => ucfirst($item->model ?? 'veo2') . ' Video',
                            ],
                        ];
                    });

                    $allVideos = $allVideos->concat($userFallVideos);
                } catch (Exception $e) {
                    Log::warning('Error fetching UserFall videos: ' . $e->getMessage());
                }
            }

            // Get ExportedVideos (with error handling)
            try {
                $exportedVideosQuery = ExportedVideo::query()
                    ->where('user_id', auth()->id())
                    ->where('status', 'completed')
                    ->whereNotNull('video_url')
                    ->where('video_url', '!=', '');

                // Apply search filter for ExportedVideos
                if (! empty($searchTerm)) {
                    $exportedVideosQuery->where('title', 'like', "%{$searchTerm}%");
                }

                $exportedVideos = $exportedVideosQuery->get()->map(function ($item) {
                    return (object) [
                        'id'          => 'exported_' . $item->id,
                        'title'       => $item->title,
                        'input'       => '',
                        'output'      => $item->video_url,
                        'output_url'  => $item->video_url,
                        'url'         => $item->video_url,
                        'created_at'  => $item->created_at,
                        'format_date' => $item->created_at->format('M d, Y'),
                        'source'      => 'exported',
                        'generator'   => (object) [
                            'id'    => null,
                            'type'  => 'video',
                            'title' => __('Exported Video'),
                        ],
                    ];
                });

                $allVideos = $allVideos->concat($exportedVideos);
            } catch (Exception $e) {
                Log::warning('Error fetching ExportedVideos: ' . $e->getMessage());
            }

            // Get uploaded video files
            $uploadedVideos = $this->getUserUploadedFiles('video');

            // Filter uploaded files by search term if provided
            if (! empty($searchTerm)) {
                $uploadedVideos = $uploadedVideos->filter(function ($item) use ($searchTerm) {
                    return Str::contains(strtolower($item->title ?? ''), strtolower($searchTerm)) ||
                        Str::contains(strtolower($item->filename ?? ''), strtolower($searchTerm));
                });
            }

            $allVideos = $allVideos->concat($uploadedVideos);

            // Apply sorting
            $allVideos = $this->sortCollection($allVideos);

            // Handle pagination if not getting all for selection
            if (! $getAllForSelection) {
                $totalCount = $allVideos->count();
                $this->hasMoreVideos = $this->loadedVideosCount < $totalCount;

                return $allVideos->take($this->loadedVideosCount);
            }

            return $allVideos;

        } catch (Exception $e) {
            Log::error('Error in getVideos(): ' . $e->getMessage());

            return collect();
        }
    }

    /**
     * Sort a collection based on current sort settings
     */
    protected function sortCollection(Collection $collection): Collection
    {
        $sortField = $this->sort;
        $direction = $this->sortAscDesc === 'desc';

        return $collection->sortBy(function ($item) use ($sortField) {
            switch ($sortField) {
                case 'title':
                    return $item->title ?? $item->filename ?? '';
                case 'input':
                    return $item->input ?? '';
                case 'created_at':
                default:
                    return $item->created_at;
            }
        }, SORT_REGULAR, $direction)->values();
    }

    public function render()
    {
        $images = collect();
        $videos = collect();
        $otherFiles = collect();
        $stockImages = collect();
        $stockVideos = collect();

        if ($this->activeFilter === 'Images') {
            $images = $this->getImages();
        } elseif ($this->activeFilter === 'Videos') {
            $videos = $this->getVideos();
        } elseif ($this->activeFilter === 'Other Files') {
            $otherFiles = $this->getOtherFiles();
        } elseif ($this->activeFilter === 'Stock Images') {
            $stockImages = $this->getStockImages();
        } elseif ($this->activeFilter === 'Stock Videos') {
            $stockVideos = $this->getStockVideos();
        }

        return view('livewire.media-manager-modal', [
            'images'      => $images,
            'videos'      => $videos,
            'otherFiles'  => $otherFiles,
            'stockImages' => $stockImages,
            'stockVideos' => $stockVideos,
        ]);
    }

    /**
     * Generate a title from a prompt by taking the first few words
     */
    private function generateTitleFromPrompt(string $prompt): string
    {
        $words = explode(' ', trim($prompt));
        $titleWords = array_slice($words, 0, 6); // Take first 6 words
        $title = implode(' ', $titleWords);

        // Add ellipsis if the prompt was longer
        if (count($words) > 6) {
            $title .= '...';
        }

        return $title ?: 'Untitled Video';
    }

    public function isCardDisabled(string $type): bool
    {
        if (in_array('all', $this->allowedTypes, true)) {
            return false;
        }

        $mapping = [
            'image' => ['image'],
            'video' => ['video'],
            'other' => ['file'],
            'file'  => ['other'],
        ];

        $allowed = array_merge(...array_values(array_intersect_key($mapping, array_flip($this->allowedTypes))));

        return ! in_array($type, $allowed, true);
    }
}
