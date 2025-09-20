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

namespace YooKassa\Request\Refunds;

use DateTime;
use YooKassa\Common\AbstractRequest;
use YooKassa\Common\Exceptions\InvalidPropertyValueTypeException;
use YooKassa\Model\Refund\RefundStatus;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель RefundsRequest.
 *
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
 * @property string $status Статус возврата
 * @property null|int $limit Ограничение количества объектов возврата, отображаемых на одной странице выдачи
 * @property string $cursor Токен для получения следующей страницы выборки
 */
class RefundsRequest extends AbstractRequest implements RefundsRequestInterface
{
    /** Максимальное количество объектов возвратов в выборке */
    public const MAX_LIMIT_VALUE = 100;

    /**
     * @var DateTime|null Время создания, от (включительно)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_createdAtGte = null;

    /**
     * @var DateTime|null Время создания, от (не включая)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_createdAtGt = null;

    /**
     * @var DateTime|null Время создания, до (включительно)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_createdAtLte = null;

    /**
     * @var DateTime|null Время создания, до (не включая)
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_createdAtLt = null;

    /**
     * @var string|null Идентификатор шлюза
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: 36)]
    #[Assert\Length(min: 36)]
    private ?string $_paymentId = null;

    /**
     * @var string|null Статус возврата
     */
    #[Assert\Choice(callback: [RefundStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_status = null;

    /**
     * @var int|null Ограничение количества объектов платежа
     */
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(value: 1)]
    #[Assert\LessThanOrEqual(self::MAX_LIMIT_VALUE)]
    private ?int $_limit = null;

    /**
     * @var string|null Токен для получения следующей страницы выборки
     */
    #[Assert\Type('string')]
    private ?string $_cursor = null;

    /**
     * Возвращает идентификатор платежа если он задан или null.
     *
     * @return null|string Идентификатор платежа
     */
    public function getPaymentId(): ?string
    {
        return $this->_paymentId;
    }

    /**
     * Проверяет, был ли задан идентификатор платежа.
     *
     * @return bool True если идентификатор был задан, false если нет
     */
    public function hasPaymentId(): bool
    {
        return !empty($this->_paymentId);
    }

    /**
     * Устанавливает идентификатор платежа или null, если требуется его удалить.
     *
     * @param null|string $value Идентификатор платежа
     *
     * @return self
     */
    public function setPaymentId(?string $value): self
    {
        $this->_paymentId = $this->validatePropertyValue('_paymentId', $value);
        return $this;
    }

    /**
     * Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCreatedAtGte(): ?DateTime
    {
        return $this->_createdAtGte;
    }

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGte(): bool
    {
        return !empty($this->_createdAtGte);
    }

    /**
     * Устанавливает дату создания от которой выбираются возвраты.
     *
     * @param DateTime|string|null $value Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGte(mixed $value): self
    {
        $this->_createdAtGte = $this->validatePropertyValue('_createdAtGte', $value);
        return $this;
    }

    /**
     * Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCreatedAtGt(): ?DateTime
    {
        return $this->_createdAtGt;
    }

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGt(): bool
    {
        return !empty($this->_createdAtGt);
    }

    /**
     * Устанавливает дату создания от которой выбираются возвраты.
     *
     * @param DateTime|string|null $value Время создания, от (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGt(mixed $value): self
    {
        $this->_createdAtGt = $this->validatePropertyValue('_createdAtGt', $value);
        return $this;
    }

    /**
     * Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCreatedAtLte(): ?DateTime
    {
        return $this->_createdAtLte;
    }

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLte(): bool
    {
        return !empty($this->_createdAtLte);
    }

    /**
     * Устанавливает дату создания до которой выбираются возвраты.
     *
     * @param DateTime|string|null $value Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtLte(mixed $value): self
    {
        $this->_createdAtLte = $this->validatePropertyValue('_createdAtLte', $value);
        return $this;
    }

    /**
     * Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCreatedAtLt(): ?DateTime
    {
        return $this->_createdAtLt;
    }

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLt(): bool
    {
        return !empty($this->_createdAtLt);
    }

    /**
     * Устанавливает дату создания до которой выбираются возвраты.
     *
     * @param DateTime|string|null $value Время создания, до (не включая) или null, чтобы удалить значение
     *
     * @return self
 */
    public function setCreatedAtLt(mixed $value): self
    {
        $this->_createdAtLt = $this->validatePropertyValue('_createdAtLt', $value);
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
     * Устанавливает статус выбираемых возвратов.
     *
     * @param string|null $value Статус выбираемых платежей или null, чтобы удалить значение
     *
     * @return self
     */
    public function setStatus(?string $value): self
    {
        $this->_status = $this->validatePropertyValue('_status', $value);
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
     * @param string|null $value Токен следующей страницы выборки или null, чтобы удалить значение
     *
     */
    public function setCursor(?string $value): self
    {
        $this->_cursor = $this->validatePropertyValue('_cursor', $value);
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
     * @param null|int|string $value Ограничение количества объектов платежа или null, чтобы удалить значение
     *
     * @throws InvalidPropertyValueTypeException Выбрасывается, если в метод было передано не целое число
     */
    public function setLimit(mixed $value): self
    {
        $this->_limit = $this->validatePropertyValue('_limit', $value);
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
     * @return RefundsRequestBuilder Билдер объектов запросов списка возвратов
     */
    public static function builder(): RefundsRequestBuilder
    {
        return new RefundsRequestBuilder();
    }
}
