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

use Exception;
use JsonException;
use Psr\Log\LoggerInterface;
use YooKassa\Common\Exceptions\ApiConnectionException;
use YooKassa\Common\Exceptions\ApiException;
use YooKassa\Common\Exceptions\AuthorizeException;
use YooKassa\Common\Exceptions\BadApiRequestException;
use YooKassa\Common\Exceptions\ExtensionNotFoundException;
use YooKassa\Common\Exceptions\ForbiddenException;
use YooKassa\Common\Exceptions\GoneException;
use YooKassa\Common\Exceptions\InternalServerError;
use YooKassa\Common\Exceptions\NotFoundException;
use YooKassa\Common\Exceptions\ResponseProcessingException;
use YooKassa\Common\Exceptions\TooManyRequestsException;
use YooKassa\Common\Exceptions\UnauthorizedException;
use YooKassa\Common\LoggerWrapper;
use YooKassa\Common\ResponseObject;
use YooKassa\Helpers\Config\ConfigurationLoader;
use YooKassa\Helpers\Config\ConfigurationLoaderInterface;
use YooKassa\Helpers\SecurityHelper;

/**
 * Класс, представляющий модель BaseClient.
 *
 * Базовый класс Curl клиента.
 *
 * @category Class
 * @package  YooKassa
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 */
class BaseClient
{
    /** Точка входа для запроса к API по магазину */
    public const ME_PATH = '/me';

    /** Точка входа для запросов к API по платежам */
    public const PAYMENTS_PATH = '/payments';

    /** Точка входа для запросов к API по возвратам */
    public const REFUNDS_PATH = '/refunds';

    /** Точка входа для запросов к API по вебхукам */
    public const WEBHOOKS_PATH = '/webhooks';

    /** Точка входа для запросов к API по чекам */
    public const RECEIPTS_PATH = '/receipts';

    /** Точка входа для запросов к API по сделкам */
    public const DEALS_PATH = '/deals';

    /** Точка входа для запросов к API по выплатам */
    public const PAYOUTS_PATH = '/payouts';

    /** Точка входа для запросов к API по персональным данным */
    public const PERSONAL_DATA_PATH = '/personal_data';

    /** Точка входа для запросов к API по участникам СБП */
    public const SBP_BANKS_PATH = '/sbp_banks';

    /** Точка входа для запросов к API по самозанятым */
    public const SELF_EMPLOYED_PATH = '/self_employed';

    /** Точка входа для запросов к API по счетам */
    public const INVOICES_PATH = '/invoices';

    /** Точка входа для запросов к API по способам оплаты */
    public const PAYMENT_METHODS_PATH = '/payment_methods';

    /** Имя HTTP заголовка, используемого для передачи idempotence key */
    public const IDEMPOTENCE_KEY_HEADER = 'Idempotence-Key';

    /**
     * Значение по умолчанию времени ожидания между запросами при отправке повторного запроса в случае получения
     * ответа с HTTP статусом 202.
     */
    public const DEFAULT_DELAY = 1800;

    /** Значение по умолчанию количества попыток получения информации от API если пришёл ответ с HTTP статусом 202 */
    public const DEFAULT_TRIES_COUNT = 3;

    /** Значение по умолчанию количества попыток получения информации от API если пришёл ответ с HTTP статусом 202 */
    public const DEFAULT_ATTEMPTS_COUNT = 3;

    /**
     * CURL клиент
     */
    protected ?ApiClientInterface $apiClient;

    /**
     * shopId магазина.
     *
     * @var int|null
     */
    protected ?int $login = null;

    /**
     * Секретный ключ магазина.
     */
    protected ?string $password = null;

    /**
     * Настройки для CURL клиента.
     */
    protected array $config = [];

    /**
     * Время через которое будут осуществляться повторные запросы.
     *
     * Значение по умолчанию - 1800 миллисекунд.
     *
     * @var int Значение в миллисекундах
     */
    protected int $timeout = 1800;

    /**
     * Количество повторных запросов при ответе API статусом 202.
     *
     * Значение по умолчанию 3
     */
    protected int $attempts = 3;

    /**
     * Объект для логирования работы SDK.
     */
    protected ?LoggerInterface $logger = null;

    /**
     * Constructor.
     */
    public function __construct(?ApiClientInterface $apiClient = null, ?ConfigurationLoaderInterface $configLoader = null)
    {
        if (null === $apiClient) {
            $apiClient = new CurlClient();
        }

        if (null === $configLoader) {
            $configLoader = new ConfigurationLoader();
        }
        $config = $configLoader->load()->getConfig();
        $this->setConfig($config);
        $apiClient->setConfig($config);

        $this->attempts = self::DEFAULT_ATTEMPTS_COUNT;
        $this->apiClient = $apiClient;
    }

    /**
     * Устанавливает авторизацию по логин/паролю.
     *
     * @example 01-client.php 7 1 Пример авторизации
     *
     * @param string|int $login
     * @param string $password
     *
     * @return $this
     */
    public function setAuth(mixed $login, string $password): self
    {
        $this->login = $login;
        $this->password = $password;

        $this->apiClient
            ->setBearerToken(null)
            ->setShopId($this->login)
            ->setShopPassword($this->password)
        ;

        return $this;
    }

    /**
     * Устанавливает авторизацию по Oauth-токену.
     *
     * @example 01-client.php 9 1 Пример авторизации
     *
     * @return $this
     */
    public function setAuthToken(string $token): self
    {
        $this->apiClient
            ->setShopId(null)
            ->setShopPassword(null)
            ->setBearerToken($token)
        ;

        return $this;
    }

    /**
     * Возвращает CURL клиента для работы с API.
     */
    public function getApiClient(): ApiClientInterface
    {
        return $this->apiClient;
    }

    /**
     * Устанавливает CURL клиента для работы с API.
     *
     * @return $this
     */
    public function setApiClient(ApiClientInterface $apiClient): self
    {
        $this->apiClient = $apiClient;
        $this->apiClient->setConfig($this->config);
        $this->apiClient->setLogger($this->logger);

        return $this;
    }

    /**
     * Устанавливает логгер приложения.
     *
     * @param null|callable|LoggerInterface|object $value Инстанс логгера
     */
    public function setLogger(mixed $value): self
    {
        if (null === $value || $value instanceof LoggerInterface) {
            $this->logger = $value;
        } else {
            $this->logger = new LoggerWrapper($value);
        }
        $this->apiClient?->setLogger($this->logger);

        return $this;
    }

    /**
     * Возвращает настройки клиента.
     */
    public function getConfig(): array
    {
        return $this->config;
    }

    /**
     * Устанавливает настройки клиента.
     */
    public function setConfig(array $config): void
    {
        $this->config = $config;
    }

    /**
     * Установка значения задержки между повторными запросами.
     *
     * @return $this
     */
    public function setRetryTimeout(int $timeout = self::DEFAULT_DELAY): self
    {
        $this->timeout = $timeout;

        return $this;
    }

    /**
     * Установка значения количества попыток повторных запросов при статусе 202.
     *
     * @return $this
     */
    public function setMaxRequestAttempts(int $attempts = self::DEFAULT_ATTEMPTS_COUNT): self
    {
        $this->attempts = $attempts;

        return $this;
    }

    /**
     * Метод проверяет, находится ли IP адрес среди IP адресов Юkassa, с которых отправляются уведомления.
     *
     * @param string $ip IPv4 или IPv6 адрес webhook уведомления
     *
     * @throws Exception Выбрасывается, если будет передан IP адрес неверного формата
     */
    public function isNotificationIPTrusted(string $ip): bool
    {
        return (new SecurityHelper())->isIPTrusted($ip);
    }

    /**
     * Кодирует массив данных в JSON строку.
     *
     * @param array $serializedData Массив данных для кодировки
     *
     * @return string Строка JSON
     *
     * @throws JsonException Выбрасывается, если не удалось конвертировать данные в строку JSON
     */
    protected function encodeData(array $serializedData): string
    {
        if ([] === $serializedData) {
            return '{}';
        }

        return json_encode($serializedData, JSON_THROW_ON_ERROR | JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    }

    /**
     * Декодирует JSON строку в массив данных.
     *
     * @param ResponseObject $response Объект ответа на запрос к API
     *
     * @return array Массив данных
     * @throws JsonException
     */
    protected function decodeData(ResponseObject $response): array
    {
        return json_decode($response->getBody(), true, 512, JSON_THROW_ON_ERROR);
    }

    /**
     * Выбрасывает исключение по коду ошибки.
     *
     * @throws ApiException неожиданный код ошибки
     * @throws BadApiRequestException Неправильный запрос. Чаще всего этот статус выдается из-за нарушения правил взаимодействия с API.
     * @throws ForbiddenException секретный ключ или OAuth-токен верный, но не хватает прав для совершения операции
     * @throws InternalServerError Технические неполадки на стороне ЮKassa. Результат обработки запроса неизвестен. Повторите запрос позднее с тем же ключом идемпотентности.
     * @throws NotFoundException ресурс не найден
     * @throws GoneException ресурс удален
     * @throws ResponseProcessingException запрос был принят на обработку, но она не завершена
     * @throws TooManyRequestsException Превышен лимит запросов в единицу времени. Попробуйте снизить интенсивность запросов.
     * @throws UnauthorizedException неверное имя пользователя или пароль или невалидный OAuth-токен при аутентификации
     * @throws AuthorizeException Ошибка авторизации. Не установлен заголовок.
     */
    protected function handleError(ResponseObject $response): void
    {
        switch ($response->getCode()) {
            case BadApiRequestException::HTTP_CODE:
                throw new BadApiRequestException($response->getHeaders(), $response->getBody());

            case ForbiddenException::HTTP_CODE:
                throw new ForbiddenException($response->getHeaders(), $response->getBody());

            case UnauthorizedException::HTTP_CODE:
                throw new UnauthorizedException($response->getHeaders(), $response->getBody());

            case InternalServerError::HTTP_CODE:
                throw new InternalServerError($response->getHeaders(), $response->getBody());

            case NotFoundException::HTTP_CODE:
                throw new NotFoundException($response->getHeaders(), $response->getBody());

            case GoneException::HTTP_CODE:
                throw new GoneException($response->getHeaders(), $response->getBody());

            case TooManyRequestsException::HTTP_CODE:
                throw new TooManyRequestsException($response->getHeaders(), $response->getBody());

            case ResponseProcessingException::HTTP_CODE:
                throw new ResponseProcessingException($response->getHeaders(), $response->getBody());

            default:
                if ($response->getCode() > 399) {
                    throw new ApiException(
                        'Unexpected response error code',
                        $response->getCode(),
                        $response->getHeaders(),
                        $response->getBody()
                    );
                }
        }
    }

    /**
     * Задержка между повторными запросами.
     *
     * @param ResponseObject $response Объект ответа на запрос к API
     * @throws JsonException
     */
    protected function delay(ResponseObject $response): void
    {
        $timeout = $this->timeout;
        $responseData = $this->decodeData($response);
        if ($timeout) {
            $delay = $timeout;
        } else {
            if (isset($responseData['retry_after'])) {
                $delay = $responseData['retry_after'];
            } else {
                $delay = self::DEFAULT_DELAY;
            }
        }
        usleep($delay * 1000);
    }

    /**
     * Выполнение запроса и обработка 202 статуса.
     *
     * @param string $path URL запроса
     * @param string $method HTTP метод
     * @param array $queryParams Массив GET параметров запроса
     * @param null|string $httpBody Тело запроса
     * @param array $headers Массив заголовков запроса
     *
     * @return mixed|ResponseObject
     *
     * @throws ApiException
     * @throws AuthorizeException
     * @throws ApiConnectionException
     * @throws ExtensionNotFoundException
     */
    protected function execute(string $path, string $method, array $queryParams, ?string $httpBody = null, array $headers = []): mixed
    {
        $attempts = $this->attempts;
        $response = $this->apiClient->call($path, $method, $queryParams, $httpBody, $headers);

        while (in_array($response->getCode(), [202, 500], true) && $attempts > 0) {
            $this->delay($response);
            --$attempts;
            $response = $this->apiClient->call($path, $method, $queryParams, $httpBody, $headers);
        }

        return $response;
    }
}
