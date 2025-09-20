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

namespace YooKassa\Request\Payments;

use DateTime;
use YooKassa\Common\AbstractRequest;
use YooKassa\Model\Payment\PaymentMethodType;
use YooKassa\Model\Payment\PaymentStatus;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель PaymentsRequest.
 *
 * Класс объекта запроса к API для получения списка платежей магазина.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property null|DateTime $createdAtGte Время создания, от (включительно)
 * @property null|DateTime $createdAtGt Время создания, от (не включая)
 * @property null|DateTime $createdAtLte Время создания, до (включительно)
 * @property null|DateTime $createdAtLt Время создания, до (не включая)
 * @property null|DateTime $capturedAtGte Время подтверждения, от (включительно)
 * @property null|DateTime $capturedAtGt Время подтверждения, от (не включая)
 * @property null|DateTime $capturedAtLte Время подтверждения, до (включительно)
 * @property null|DateTime $capturedAtLt Время подтверждения, до (не включая)
 * @property null|string $status Статус платежа
 * @property null|string $paymentMethod Платежный метод
 * @property null|int $limit Ограничение количества объектов платежа, отображаемых на одной странице выдачи
 * @property null|string $cursor Страница выдачи результатов, которую необходимо отобразить
 */
class PaymentsRequest extends AbstractRequest implements PaymentsRequestInterface
{
    /** Максимальное количество объектов платежа в выборке */
    public const MAX_LIMIT_VALUE = 100;

    /**
     * @var DateTime|null Время создания, от (включительно)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_gte = null;

    /**
     * @var DateTime|null Время создания, от (не включая)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_gt = null;

    /**
     * @var DateTime|null Время создания, до (включительно)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_lte = null;

    /**
     * @var DateTime|null Время создания, до (не включая)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_lt = null;

    /**
     * @var DateTime|null Время подтверждения, от (включительно)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_captured_at_gte = null;

    /**
     * @var DateTime|null Время подтверждения, от (не включая)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_captured_at_gt = null;

    /**
     * @var DateTime|null Время подтверждения, до (включительно)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_captured_at_lte = null;

    /**
     * @var DateTime|null Время подтверждения, до (не включая)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_captured_at_lt = null;

    /**
     * @var string|null Статус платежа
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentStatus::class, 'getValidValues'])]
    private ?string $_status = null;

    /**
     * @var string|null Платежный метод
     */
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [PaymentMethodType::class, 'getValidValues'])]
    private ?string $_payment_method = null;

    /**
     * @var null|int Ограничение количества объектов платежа
     */
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(value: 1)]
    #[Assert\LessThanOrEqual(value: self::MAX_LIMIT_VALUE)]
    private ?int $_limit = null;

    /**
     * @var string|null Страница выдачи результатов, которую необходимо отобразить
     */
    #[Assert\Type('string')]
    private ?string $_cursor = null;

    /**
     * Возвращает дату создания от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCreatedAtGte(): ?DateTime
    {
        return $this->_created_at_gte;
    }

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGte(): bool
    {
        return null !== $this->_created_at_gte;
    }

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param DateTime|string|null $_created_at_gte Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGte(mixed $_created_at_gte = null): self
    {
        $this->_created_at_gte = $this->validatePropertyValue('_created_at_gte', $_created_at_gte);
        return $this;
    }

    /**
     * Возвращает дату создания от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCreatedAtGt(): ?DateTime
    {
        return $this->_created_at_gt;
    }

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGt(): bool
    {
        return null !== $this->_created_at_gt;
    }

    /**
     * Устанавливает дату создания от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $created_at_gt Время создания, от (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGt(mixed $created_at_gt): self
    {
        $this->_created_at_gt = $this->validatePropertyValue('_created_at_gt', $created_at_gt);
        return $this;
    }

    /**
     * Возвращает дату создания до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCreatedAtLte(): ?DateTime
    {
        return $this->_created_at_lte;
    }

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLte(): bool
    {
        return null !== $this->_created_at_lte;
    }

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $created_at_lte Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtLte(mixed $created_at_lte): self
    {
        $this->_created_at_lte = $this->validatePropertyValue('_created_at_lte', $created_at_lte);
        return $this;
    }

    /**
     * Возвращает дату создания до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCreatedAtLt(): ?DateTime
    {
        return $this->_created_at_lt;
    }

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLt(): bool
    {
        return null !== $this->_created_at_lt;
    }

    /**
     * Устанавливает дату создания до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $created_at_lt Время создания, до (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtLt(mixed $created_at_lt): self
    {
        $this->_created_at_lt = $this->validatePropertyValue('_created_at_lt', $created_at_lt);
        return $this;
    }

    /**
     * Возвращает дату подтверждения от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время подтверждения, от (включительно)
     */
    public function getCapturedAtGte(): ?DateTime
    {
        return $this->_captured_at_gte;
    }

    /**
     * Проверяет, была ли установлена дата подтверждения от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtGte(): bool
    {
        return null !== $this->_captured_at_gte;
    }

    /**
     * Устанавливает дату подтверждения от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $captured_at_gte Время подтверждения, от (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCapturedAtGte(mixed $captured_at_gte): self
    {
        $this->_captured_at_gte = $this->validatePropertyValue('_captured_at_gte', $captured_at_gte);
        return $this;
    }

    /**
     * Возвращает дату подтверждения от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время подтверждения, от (не включая)
     */
    public function getCapturedAtGt(): ?DateTime
    {
        return $this->_captured_at_gt;
    }

    /**
     * Проверяет, была ли установлена дата подтверждения от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtGt(): bool
    {
        return null !== $this->_captured_at_gt;
    }

    /**
     * Устанавливает дату подтверждения от которой выбираются платежи.
     *
     * @param null|DateTime|int|string $captured_at_gt Время подтверждения, от (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCapturedAtGt(mixed $captured_at_gt): self
    {
        $this->_captured_at_gt = $this->validatePropertyValue('_captured_at_gt', $captured_at_gt);
        return $this;
    }

    /**
     * Возвращает дату подтверждения до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время подтверждения, до (включительно)
     */
    public function getCapturedAtLte(): ?DateTime
    {
        return $this->_captured_at_lte;
    }

    /**
     * Проверяет, была ли установлена дата подтверждения до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtLte(): bool
    {
        return null !== $this->_captured_at_lte;
    }

    /**
     * Устанавливает дату подтверждения до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $captured_at_lte Время подтверждения, до (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCapturedAtLte(mixed $captured_at_lte): self
    {
        $this->_captured_at_lte = $this->validatePropertyValue('_captured_at_lte', $captured_at_lte);
        return $this;
    }

    /**
     * Возвращает дату подтверждения до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время подтверждения, до (не включая)
     */
    public function getCapturedAtLt(): ?DateTime
    {
        return $this->_captured_at_lt;
    }

    /**
     * Проверяет, была ли установлена дата подтверждения до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCapturedAtLt(): bool
    {
        return null !== $this->_captured_at_lt;
    }

    /**
     * Устанавливает дату подтверждения до которой выбираются платежи.
     *
     * @param null|DateTime|int|string $captured_at_lt Время подтверждения, до (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCapturedAtLt(mixed $captured_at_lt): self
    {
        $this->_captured_at_lt = $this->validatePropertyValue('_captured_at_lt', $captured_at_lt);
        return $this;
    }

    /**
     * Возвращает статус выбираемых платежей или null, если он до этого не был установлен.
     *
     * @return null|string Статус выбираемых платежей
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Проверяет, был ли установлен статус выбираемых платежей.
     *
     * @return bool True если статус был установлен, false если нет
     */
    public function hasStatus(): bool
    {
        return null !== $this->_status;
    }

    /**
     * Устанавливает статус выбираемых платежей.
     *
     * @param string|null $status Статус выбираемых платежей или null, чтобы удалить значение
     *
     * @return self
     */
    public function setStatus(?string $status): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает платежный метод выбираемых платежей или null, если он до этого не был установлен.
     *
     * @return null|string Платежный метод выбираемых платежей
     */
    public function getPaymentMethod(): ?string
    {
        return $this->_payment_method;
    }

    /**
     * Проверяет, был ли установлен платежный метод выбираемых платежей.
     *
     * @return bool True если платежный метод был установлен, false если нет
     */
    public function hasPaymentMethod(): bool
    {
        return null !== $this->_payment_method;
    }

    /**
     * Устанавливает платежный метод выбираемых платежей.
     *
     * @param string|null $payment_method Платежный метод выбираемых платежей или null, чтобы удалить значение
     *
     * @return self
     */
    public function setPaymentMethod(?string $payment_method): self
    {
        $this->_payment_method = $this->validatePropertyValue('_payment_method', $payment_method);
        return $this;
    }

    /**
     * Ограничение количества объектов платежа.
     *
     * @return null|int Ограничение количества объектов платежа
     */
    public function getLimit(): ?int
    {
        return $this->_limit;
    }

    /**
     * Проверяет, было ли установлено ограничение количества объектов платежа.
     *
     * @return bool True если было установлено, false если нет
     */
    public function hasLimit(): bool
    {
        return null !== $this->_limit;
    }

    /**
     * Устанавливает ограничение количества объектов платежа.
     *
     * @param null|int|string $limit Ограничение количества объектов платежа или null, чтобы удалить значение
     *
     * @return self
     */
    public function setLimit(mixed $limit): self
    {
        $this->_limit = $this->validatePropertyValue('_limit', $limit);
        return $this;
    }

    /**
     * Страница выдачи результатов, которую необходимо отобразить.
     */
    public function getCursor(): ?string
    {
        return $this->_cursor;
    }

    /**
     * Проверяет, была ли установлена страница выдачи результатов, которую необходимо отобразить.
     *
     * @return bool True если была установлена, false если нет
     */
    public function hasCursor(): bool
    {
        return null !== $this->_cursor;
    }

    /**
     * Устанавливает страницу выдачи результатов, которую необходимо отобразить
     *
     * @param string|null $cursor Страница выдачи результатов или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCursor(?string $cursor): self
    {
        $this->_cursor = $this->validatePropertyValue('_cursor', $cursor);
        return $this;
    }

    /**
     * Проверяет валидность текущего объекта запроса.
     *
     * @return bool True если объект валиден, false если нет
     */
    public function validate(): bool
    {
        return true;
    }

    /**
     * Возвращает инстанс билдера объектов запросов списка платежей магазина.
     *
     * @return PaymentsRequestBuilder Билдер объектов запросов списка платежей
     */
    public static function builder(): PaymentsRequestBuilder
    {
        return new PaymentsRequestBuilder();
    }
}
