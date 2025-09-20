<?php

/*
 * The MIT License
 *
 * Copyright (c) 2025 "YooMoney", NBСO LLC
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

namespace YooKassa\Client;

use CurlHandle;
use Psr\Log\LoggerInterface;
use YooKassa\Common\Exceptions\ApiConnectionException;
use YooKassa\Common\Exceptions\ApiException;
use YooKassa\Common\Exceptions\AuthorizeException;
use YooKassa\Common\Exceptions\ExtensionNotFoundException;
use YooKassa\Common\ResponseObject;
use YooKassa\Helpers\RawHeadersParser;

/**
 * Класс, представляющий модель CurlClient.
 *
 * Класс клиента Curl запросов.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class CurlClient implements ApiClientInterface
{
    /** @var array Настройки клиента */
    private array $config = [];

    /** @var int|null shopId магазина */
    private ?int $shopId = null;

    /** @var string|null Секретный ключ магазина */
    private ?string $shopPassword = null;

    /** @var null|string OAuth токен */
    private ?string $bearerToken = null;

    /** @var int Настройка параметра CURLOPT_TIMEOUT */
    private int $timeout = 80;

    /** @var int Настройка параметра CURLOPT_CONNECTTIMEOUT */
    private int $connectionTimeout = 30;

    /** @var null|string Настройка прокси-сервера, если нужен */
    private ?string $proxy = null;

    /** @var UserAgent Строка user-agent для статистики */
    private UserAgent $userAgent;

    /** @var bool Настройка удержания соединения */
    private bool $keepAlive = true;

    /** @var array Заголовки по умолчанию */
    private array $defaultHeaders = [
        'Content-Type' => 'application/json',
        'Accept' => 'application/json',
    ];

    /** @var CurlHandle|null Текущий ресурс для работы с curl */
    private ?CurlHandle $curl = null;

    /** @var null|LoggerInterface Объект для логирования запросов */
    private ?LoggerInterface $logger = null;

    /**
     * CurlClient constructor.
     */
    public function __construct()
    {
        $this->userAgent = new UserAgent();
    }

    public function setLogger(?LoggerInterface $logger): void
    {
        $this->logger = $logger;
    }

    /**
     * {@inheritdoc}
     *
     * @param string $path URL запроса
     * @param string $method HTTP метод
     * @param array $queryParams Массив GET параметров запроса
     * @param null|string $httpBody Тело запроса
     * @param array $headers Массив заголовков запроса
     *
     * @throws ApiConnectionException
     * @throws ApiException
     * @throws AuthorizeException
     * @throws ExtensionNotFoundException
     */
    public function call(string $path, string $method, array $queryParams, ?string $httpBody = null, array $headers = []): ResponseObject
    {
        $headers = $this->prepareHeaders($headers);

        $this->logRequestParams($path, $method, $queryParams, $httpBody, $headers);

        $url = $this->prepareUrl($path, $queryParams);

        $this->prepareCurl($method, $url, $httpBody, $this->implodeHeaders($headers));

        [$httpHeaders, $httpBody, $responseInfo] = $this->sendRequest();

        if (!$this->keepAlive) {
            $this->closeCurlConnection();
        }

        $this->logResponse($httpBody, $responseInfo, $httpHeaders);

        return new ResponseObject([
            'code' => $responseInfo['http_code'],
            'headers' => $httpHeaders,
            'body' => $httpBody,
        ]);
    }

    /**
     * Устанавливает параметры CURL.
     *
     * @param string $optionName Имя параметра
     * @param mixed $optionValue Значение параметра
     */
    public function setCurlOption(string $optionName, mixed $optionValue): bool
    {
        return curl_setopt($this->curl, $optionName, $optionValue);
    }

    /**
     * Close connection.
     */
    public function closeCurlConnection(): void
    {
        if (null !== $this->curl) {
            curl_close($this->curl);
        }
    }

    /**
     * Выполняет запрос, получает и возвращает обработанный ответ
     *
     * @throws ApiConnectionException
     */
    public function sendRequest(): array
    {
        $response = curl_exec($this->curl);
        $httpHeaderSize = curl_getinfo($this->curl, CURLINFO_HEADER_SIZE);
        $httpHeaders = RawHeadersParser::parse(substr($response, 0, $httpHeaderSize));
        $httpBody = substr($response, $httpHeaderSize);
        $responseInfo = curl_getinfo($this->curl);
        $curlError = curl_error($this->curl);
        $curlErrno = curl_errno($this->curl);
        if (false === $response) {
            $this->handleCurlError($curlError, $curlErrno);
        }

        return [$httpHeaders, $httpBody, $responseInfo];
    }

    /**
     * Устанавливает тело запроса.
     *
     * @param string $method HTTP метод
     * @param string|null $httpBody Тело запроса
     */
    public function setBody(string $method, ?string $httpBody = null): void
    {
        $this->setCurlOption(CURLOPT_CUSTOMREQUEST, $method);
        if (!empty($httpBody)) {
            $this->setCurlOption(CURLOPT_POSTFIELDS, $httpBody);
        }
    }

    /**
     * Устанавливает shopId магазина.
     *
     * @param mixed $shopId shopId магазина
     *
     * @return $this
     */
    public function setShopId(mixed $shopId): self
    {
        $this->shopId = (int) $shopId;

        return $this;
    }

    /**
     * Устанавливает секретный ключ магазина.
     *
     * @param string|null $shopPassword Секретный ключ магазина
     *
     * @return $this
     */
    public function setShopPassword(?string $shopPassword): self
    {
        $this->shopPassword = $shopPassword;

        return $this;
    }

    /**
     * Возвращает значение параметра CURLOPT_TIMEOUT.
     */
    public function getTimeout(): int
    {
        return $this->timeout;
    }

    /**
     * Устанавливает значение параметра CURLOPT_TIMEOUT.
     *
     * @param int $timeout Максимальное количество секунд для выполнения функций cURL
     */
    public function setTimeout(int $timeout): void
    {
        $this->timeout = $timeout;
    }

    /**
     * Возвращает значение параметра CURLOPT_CONNECTTIMEOUT.
     */
    public function getConnectionTimeout(): int
    {
        return $this->connectionTimeout;
    }

    /**
     * Устанавливает значение параметра CURLOPT_CONNECTTIMEOUT.
     *
     * @param int $connectionTimeout Число секунд ожидания при попытке подключения
     */
    public function setConnectionTimeout(int $connectionTimeout = 30): void
    {
        $this->connectionTimeout = $connectionTimeout;
    }

    /**
     * Возвращает настройки прокси.
     */
    public function getProxy(): string
    {
        return $this->proxy;
    }

    /**
     * Устанавливает настройки прокси.
     *
     * @param string $proxy Прокси сервер
     */
    public function setProxy(string $proxy): void
    {
        $this->proxy = $proxy;
    }

    /**
     * Возвращает настройки.
     */
    public function getConfig(): array
    {
        return $this->config;
    }

    /**
     * Устанавливает настройки.
     *
     * @param array $config Настройки клиента
     */
    public function setConfig(array $config): void
    {
        $this->config = $config;
    }

    /**
     * Возвращает UserAgent.
     */
    public function getUserAgent(): UserAgent
    {
        return $this->userAgent;
    }

    /**
     * Устанавливает OAuth-токен магазина.
     *
     * @param null|string $bearerToken OAuth-токен магазина
     *
     * @return $this
     */
    public function setBearerToken(?string $bearerToken): self
    {
        $this->bearerToken = $bearerToken;

        return $this;
    }

    /**
     * Устанавливает флаг сохранения соединения.
     *
     * @param bool $keepAlive Флаг сохранения настроек
     *
     * @return $this
     */
    public function setKeepAlive(bool $keepAlive): self
    {
        $this->keepAlive = $keepAlive;

        return $this;
    }

    public function setAdvancedCurlOptions(): void
    {
    }

    /**
     * @return void
     *
     * @throws ExtensionNotFoundException
     */
    private function initCurl(): void
    {
        if (!extension_loaded('curl')) {
            throw new ExtensionNotFoundException('curl');
        }

        if (!$this->curl || !$this->keepAlive) {
            $this->curl = curl_init();
        }

    }

    /**
     * @throws ApiConnectionException
     */
    private function handleCurlError(string $error, int $errno): void
    {
        $msg = match ($errno) {
            CURLE_COULDNT_CONNECT, CURLE_COULDNT_RESOLVE_HOST, CURLE_OPERATION_TIMEOUTED => 'Could not connect to YooKassa API. Please check your internet connection and try again.',
            CURLE_SSL_CACERT, CURLE_SSL_PEER_CERTIFICATE => 'Could not verify SSL certificate.',
            default => 'Unexpected error communicating.',
        };
        $msg .= sprintf("\n\n(Network error [errno %s]: %s)", $errno, $error);

        throw new ApiConnectionException($msg);
    }

    /**
     * Возвращает базовый URL для API Кассы
     *
     * @return string
     */
    private function getUrl(): string
    {
        $config = $this->config;

        return (string) $config['url'];
    }

    /**
     * @throws AuthorizeException
     */
    private function prepareHeaders(array $headers): array
    {
        $headers = array_merge($this->defaultHeaders, $headers);

        $headers[UserAgent::HEADER] = $this->getUserAgent()->getHeaderString();

        if ($this->shopId && $this->shopPassword) {
            $encodedAuth = base64_encode($this->shopId . ':' . $this->shopPassword);
            $headers['Authorization'] = 'Basic ' . $encodedAuth;
        } elseif ($this->bearerToken) {
            $headers['Authorization'] = 'Bearer ' . $this->bearerToken;
        }

        if (empty($headers['Authorization'])) {
            throw new AuthorizeException('Authorization headers not set');
        }

        return $headers;
    }

    private function implodeHeaders(array $headers): array
    {
        return array_map(static fn ($key, $value) => $key . ':' . $value, array_keys($headers), $headers);
    }

    private function logRequestParams(string $path, string $method, array $queryParams = [], ?string $httpBody = null, array $headers = []): void
    {
        if (null !== $this->logger) {
            $message = 'Send request: ' . $method . ' ' . $path;
            $context = [];
            if (!empty($queryParams)) {
                $context['_params'] = $queryParams;
            }
            if (!empty($httpBody)) {
                $data = json_decode($httpBody, true);
                if (JSON_ERROR_NONE !== json_last_error()) {
                    $data = $httpBody;
                }
                $context['_body'] = $data;
            }
            if (!empty($headers)) {
                $context['_headers'] = $headers;
            }
            $this->logger->info($message, $context);
        }
    }

    private function prepareUrl(string $path, array $queryParams): string
    {
        $url = $this->getUrl() . $path;

        if (!empty($queryParams)) {
            $url .= '?' . http_build_query($queryParams);
        }

        return $url;
    }

    private function logResponse(string $httpBody, array $responseInfo, array $httpHeaders): void
    {
        if (null !== $this->logger) {
            $message = 'Response with code ' . $responseInfo['http_code'] . ' received.';
            $context = [];
            if (!empty($httpBody)) {
                $data = json_decode($httpBody, true);
                if (JSON_ERROR_NONE !== json_last_error()) {
                    $data = $httpBody;
                }
                $context['_body'] = $data;
            }
            if (!empty($httpHeaders)) {
                $context['_headers'] = $httpHeaders;
            }
            $this->logger->info($message, $context);
        }
    }

    /**
     * @throws ExtensionNotFoundException
     */
    private function prepareCurl(string $method, string $url, ?string $httpBody = null, array $headers = []): void
    {
        $this->initCurl();

        $this->setCurlOption(CURLOPT_URL, $url);

        $this->setCurlOption(CURLOPT_RETURNTRANSFER, true);

        $this->setCurlOption(CURLOPT_HEADER, true);

        if ($this->proxy) {
            $this->setCurlOption(CURLOPT_PROXY, $this->proxy);
            $this->setCurlOption(CURLOPT_HTTPPROXYTUNNEL, true);
        }

        $this->setBody($method, $httpBody);

        $this->setCurlOption(CURLOPT_HTTPHEADER, $headers);

        $this->setCurlOption(CURLOPT_CONNECTTIMEOUT, $this->connectionTimeout);

        $this->setCurlOption(CURLOPT_TIMEOUT, $this->timeout);

        $this->setAdvancedCurlOptions();
    }
}
