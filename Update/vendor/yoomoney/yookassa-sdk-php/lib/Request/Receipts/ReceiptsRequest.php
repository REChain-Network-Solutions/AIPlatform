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

namespace YooKassa\Request\Receipts;

use DateTime;
use YooKassa\Common\AbstractRequest;
use YooKassa\Model\Payment\ReceiptRegistrationStatus;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс объекта запроса к API списка возвратов магазина.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property DateTime $createdAtGte Время создания, от (включительно)
 * @property DateTime $created_at_gte Время создания, от (включительно)
 * @property DateTime $createdAtGt Время создания, от (не включая)
 * @property DateTime $created_at_gt Время создания, от (не включая)
 * @property DateTime $createdAtLte Время создания, до (включительно)
 * @property DateTime $created_at_lte Время создания, до (включительно)
 * @property DateTime $createdAtLt Время создания, до (не включая)
 * @property DateTime $created_at_lt Время создания, до (не включая)
 * @property string $paymentId Идентификатор платежа
 * @property string $payment_id Идентификатор платежа
 * @property string $refundId Идентификатор возврата
 * @property string $refund_id Идентификатор возврата
 * @property string $status Статус возврата
 * @property null|int $limit Ограничение количества объектов возврата, отображаемых на одной странице выдачи
 * @property string $cursor Токен для получения следующей страницы выборки
 */
class ReceiptsRequest extends AbstractRequest implements ReceiptsRequestInterface
{
    /** Максимальное количество объектов чеков в выборке */
    public const MAX_LIMIT_VALUE = 100;

    /** Длина идентификатора платежа */
    public const LENGTH_PAYMENT_ID = 36;

    /** Длина идентификатора платежа */
    public const LENGTH_REFUND_ID = 36;

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
     * @var string|null Идентификатор платежа
     */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::LENGTH_PAYMENT_ID)]
    private ?string $_payment_id = null;

    /**
     * @var string|null Идентификатор возврата
     */
    #[Assert\Type('string')]
    #[Assert\Length(exactly: self::LENGTH_REFUND_ID)]
    private ?string $_refund_id = null;

    /**
     * @var string|null Статус чека
     */
    #[Assert\NotBlank]
    #[Assert\Type('string')]
    #[Assert\Choice(callback: [ReceiptRegistrationStatus::class, 'getValidValues'])]
    private ?string $_status = null;

    /**
     * @var null|int Ограничение количества объектов платежа
     */
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(value: 1)]
    #[Assert\LessThanOrEqual(value: self::MAX_LIMIT_VALUE)]
    private ?int $_limit = null;

    /**
     * @var string|null Токен для получения следующей страницы выборки
     */
    #[Assert\Type('string')]
    private ?string $_cursor = null;

    /**
     * Возвращает идентификатор возврата.
     *
     * @return string|null Идентификатор возврата
     */
    public function getRefundId(): ?string
    {
        return $this->_refund_id;
    }

    /**
     * Проверяет, был ли установлен идентификатор возврата.
     *
     * @return bool True если идентификатор возврата был установлен, false если не был
     */
    public function hasRefundId(): bool
    {
        return !empty($this->_refund_id);
    }

    /**
     * Устанавливает идентификатор возврата.
     *
     * @param string|null $refund_id Идентификатор возврата, который ищется в API
     *
     * @return self
     */
    public function setRefundId(?string $refund_id): self
    {
        $this->_refund_id = $this->validatePropertyValue('_refund_id', $refund_id);
        return $this;
    }

    /**
     * Возвращает идентификатор платежа если он задан или null.
     *
     * @return null|string Идентификатор платежа
     */
    public function getPaymentId(): ?string
    {
        return $this->_payment_id;
    }

    /**
     * Проверяет, был ли задан идентификатор платежа.
     *
     * @return bool True если идентификатор был задан, false если нет
     */
    public function hasPaymentId(): bool
    {
        return !empty($this->_payment_id);
    }

    /**
     * Устанавливает идентификатор платежа или null, если требуется его удалить.
     *
     * @param null|string $payment_id Идентификатор платежа
     */
    public function setPaymentId(?string $payment_id): self
    {
        $this->_payment_id = $this->validatePropertyValue('_payment_id', $payment_id);
        return $this;
    }

    /**
     * Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCreatedAtGte(): ?DateTime
    {
        return $this->_created_at_gte;
    }

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGte(): bool
    {
        return !empty($this->_created_at_gte);
    }

    /**
     * Устанавливает дату создания от которой выбираются возвраты.
     *
     * @param null|DateTime|int|string $_created_at_gte Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGte(mixed $_created_at_gte): self
    {
        $this->_created_at_gte = $this->validatePropertyValue('_created_at_gte', $_created_at_gte);
        return $this;
    }

    /**
     * Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCreatedAtGt(): ?DateTime
    {
        return $this->_created_at_gt;
    }

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGt(): bool
    {
        return !empty($this->_created_at_gt);
    }

    /**
     * Устанавливает дату создания от которой выбираются возвраты.
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
     * Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCreatedAtLte(): ?DateTime
    {
        return $this->_created_at_lte;
    }

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLte(): bool
    {
        return !empty($this->_created_at_lte);
    }

    /**
     * Устанавливает дату создания до которой выбираются возвраты.
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
     * Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCreatedAtLt(): ?DateTime
    {
        return $this->_created_at_lt;
    }

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLt(): bool
    {
        return !empty($this->_created_at_lt);
    }

    /**
     * Устанавливает дату создания до которой выбираются возвраты.
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
     * Возвращает статус выбираемых возвратов или null, если он до этого не был установлен.
     *
     * @return null|string Статус выбираемых возвратов
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Проверяет, был ли установлен статус выбираемых возвратов.
     *
     * @return bool True если статус был установлен, false если нет
     */
    public function hasStatus(): bool
    {
        return !empty($this->_status);
    }

    /**
     * Устанавливает статус выбираемых чеков
     *
     * @param string|null $status Статус выбираемых чеков или null, чтобы удалить значение
     *
     * @return self
     */
    public function setStatus(?string $status): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает токен для получения следующей страницы выборки.
     *
     * @return null|string Токен для получения следующей страницы выборки
     */
    public function getCursor(): ?string
    {
        return $this->_cursor;
    }

    /**
     * Проверяет, был ли установлен токен следующей страницы.
     *
     * @return bool True если токен был установлен, false если нет
     */
    public function hasCursor(): bool
    {
        return !empty($this->_cursor);
    }

    /**
     * Устанавливает токен следующей страницы выборки.
     *
     * @param string|null $cursor Токен следующей страницы выборки или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCursor(?string $cursor): self
    {
        $this->_cursor = $this->validatePropertyValue('_cursor', $cursor);
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
     * @param null|int $limit Ограничение количества объектов платежа или null, чтобы удалить значение
     *
     * @return self
     */
    public function setLimit(?int $limit): self
    {
        $this->_limit = $this->validatePropertyValue('_limit', $limit);
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
     * Возвращает инстанс билдера объектов запросов списка возвратов магазина.
     *
     * @return ReceiptsRequestBuilder Билдер объектов запросов списка возвратов
     */
    public static function builder(): ReceiptsRequestBuilder
    {
        return new ReceiptsRequestBuilder();
    }
}
