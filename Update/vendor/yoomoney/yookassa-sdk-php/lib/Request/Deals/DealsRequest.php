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

namespace YooKassa\Request\Deals;

use DateTime;
use YooKassa\Common\AbstractRequest;
use YooKassa\Model\Deal\DealStatus;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель DealsRequest.
 * Класс объекта запроса к API для получения списка сделок магазина.
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property null|string $cursor Страница выдачи результатов, которую необходимо отобразить
 * @property null|int $limit Ограничение количества объектов платежа, отображаемых на одной странице выдачи
 * @property null|DateTime $createdAtGte Время создания, от (включительно)
 * @property null|DateTime $created_at_gte Время создания, от (включительно)
 * @property null|DateTime $createdAtGt Время создания, от (не включая)
 * @property null|DateTime $created_at_gt Время создания, от (не включая)
 * @property null|DateTime $createdAtLte Время создания, до (включительно)
 * @property null|DateTime $created_at_lte Время создания, до (включительно)
 * @property null|DateTime $createdAtLt Время создания, до (не включая)
 * @property null|DateTime $created_at_lt Время создания, до (не включая)
 * @property null|DateTime $expiresAtGte Время автоматического закрытия, от (включительно)
 * @property null|DateTime $expires_at_gte Время автоматического закрытия, от (включительно)
 * @property null|DateTime $expiresAtGt Время автоматического закрытия, от (не включая)
 * @property null|DateTime $expires_at_gt Время автоматического закрытия, от (не включая)
 * @property null|DateTime $expiresAtLte Время автоматического закрытия, до (включительно)
 * @property null|DateTime $expires_at_lte Время автоматического закрытия, до (включительно)
 * @property null|DateTime $expiresAtLt Время автоматического закрытия, до (не включая)
 * @property null|DateTime $expires_at_lt Время автоматического закрытия, до (не включая)
 * @property null|string $fullTextSearch Фильтр по описанию сделки — параметру description
 * @property null|string $full_text_search Фильтр по описанию сделки — параметру description
 * @property null|string $status Статус платежа
 */
class DealsRequest extends AbstractRequest implements DealsRequestInterface
{
    /** @var int Максимальное количество объектов платежа в выборке */
    public const MAX_LIMIT_VALUE = 100;

    /** @var int Максимальное количество символов для поиска */
    public const MAX_LENGTH_DESCRIPTION = 128;

    /** @var int Минимальное количество символов для поиска */
    public const MIN_LENGTH_DESCRIPTION = 4;

    /**
     * Время создания, от (включительно).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_gte = null;

    /**
     * Время создания, от (не включая).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_gt = null;

    /**
     * Время создания, до (включительно).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_lte = null;

    /**
     * Время создания, до (не включая).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_created_at_lt = null;

    /**
     * Время автоматического закрытия, от (включительно).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_expires_at_gte = null;

    /**
     * Время автоматического закрытия, от (не включая).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_expires_at_gt = null;

    /**
     * Время автоматического закрытия, до (включительно).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_expires_at_lte = null;

    /**
     * Время автоматического закрытия, до (не включая).
     *
     * @var DateTime|null
     */
    #[Assert\DateTime(format: YOOKASSA_DATE)]
    #[Assert\Type('DateTime')]
    private ?DateTime $_expires_at_lt = null;

    /**
     * Статус сделки
     *
     * @var string|null
     */
    #[Assert\Choice(callback: [DealStatus::class, 'getValidValues'])]
    #[Assert\Type('string')]
    private ?string $_status = null;

    /**
     * Фильтр по описанию сделки — параметру description. От 4 до 128 символов.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    #[Assert\Length(max: self::MAX_LENGTH_DESCRIPTION)]
    #[Assert\Length(min: self::MIN_LENGTH_DESCRIPTION)]
    private ?string $_full_text_search = null;

    /**
     * Размер выдачи результатов запроса — количество объектов, передаваемых в ответе. Возможные значения: от 1 до 100. Пример: limit=50
     *
     * @var int|null
     */
    #[Assert\NotNull]
    #[Assert\Type('int')]
    #[Assert\GreaterThanOrEqual(value: 1)]
    #[Assert\LessThanOrEqual(self::MAX_LIMIT_VALUE)]
    private ?int $_limit = 10;

    /**
     * Указатель на следующий фрагмент списка. Пример: cursor=37a5c87d-3984-51e8-a7f3-8de646d39ec15.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_cursor = null;

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
     * Страница выдачи результатов, которую необходимо отобразить.
     *
     * @return string|null
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
     * Устанавливает страницу выдачи результатов, которую необходимо отобразить.
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
     * @param DateTime|string|null $created_at_gte Время создания, от (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGte(DateTime|string|null $created_at_gte): self
    {
        $this->_created_at_gte = $this->validatePropertyValue('_created_at_gte', $created_at_gte);
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
     * @param DateTime|string|null $created_at_gt Время создания, от (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtGt(DateTime|string|null $created_at_gt): self
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
     * @param DateTime|string|null $created_at_lte Время создания, до (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtLte(DateTime|string|null $created_at_lte): self
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
     * @param DateTime|string|null $created_at_lt Время создания, до (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setCreatedAtLt(DateTime|string|null $created_at_lt): self
    {
        $this->_created_at_lt = $this->validatePropertyValue('_created_at_lt', $created_at_lt);
        return $this;
    }

    /**
     * Возвращает дату автоматического закрытия от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, от (включительно)
     */
    public function getExpiresAtGte(): ?DateTime
    {
        return $this->_expires_at_gte;
    }

    /**
     * Проверяет, была ли установлена дата автоматического закрытия от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtGte(): bool
    {
        return null !== $this->_expires_at_gte;
    }

    /**
     * Устанавливает дату автоматического закрытия от которой выбираются платежи.
     *
     * @param DateTime|string|null $expires_at_gte Время автоматического закрытия, от (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setExpiresAtGte(DateTime|string|null $expires_at_gte): self
    {
        $this->_expires_at_gte = $this->validatePropertyValue('_expires_at_gte', $expires_at_gte);
        return $this;
    }

    /**
     * Возвращает дату автоматического закрытия от которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, от (не включая)
     */
    public function getExpiresAtGt(): ?DateTime
    {
        return $this->_expires_at_gt;
    }

    /**
     * Проверяет, была ли установлена дата автоматического закрытия от которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtGt(): bool
    {
        return null !== $this->_expires_at_gt;
    }

    /**
     * Устанавливает дату автоматического закрытия от которой выбираются платежи.
     *
     * @param DateTime|string|null $expires_at_lt Время автоматического закрытия, от (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setExpiresAtGt(DateTime|string|null $expires_at_lt): self
    {
        $this->_expires_at_gt = $this->validatePropertyValue('_expires_at_gt', $expires_at_lt);
        return $this;
    }

    /**
     * Возвращает дату автоматического закрытия до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, до (включительно)
     */
    public function getExpiresAtLte(): ?DateTime
    {
        return $this->_expires_at_lte;
    }

    /**
     * Проверяет, была ли установлена дата автоматического закрытия до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtLte(): bool
    {
        return null !== $this->_expires_at_lte;
    }

    /**
     * Устанавливает дату автоматического закрытия до которой выбираются платежи.
     *
     * @param DateTime|string|null $expires_at_lte Время автоматического закрытия, до (включительно) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setExpiresAtLte(DateTime|string|null $expires_at_lte): self
    {
        $this->_expires_at_lte = $this->validatePropertyValue('_expires_at_lte', $expires_at_lte);
        return $this;
    }

    /**
     * Возвращает дату автоматического закрытия до которой будут возвращены платежи или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, до (не включая)
     */
    public function getExpiresAtLt(): ?DateTime
    {
        return $this->_expires_at_lt;
    }

    /**
     * Проверяет, была ли установлена дата автоматического закрытия до которой выбираются платежи.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtLt(): bool
    {
        return null !== $this->_expires_at_lt;
    }

    /**
     * Устанавливает дату автоматического закрытия до которой выбираются платежи.
     *
     * @param DateTime|string|null $expires_at_lt Время автоматического закрытия, до (не включая) или null, чтобы удалить значение
     *
     * @return self
     */
    public function setExpiresAtLt(DateTime|string|null $expires_at_lt): self
    {
        $this->_expires_at_lt = $this->validatePropertyValue('_expires_at_lt', $expires_at_lt);
        return $this;
    }

    /**
     * Возвращает статус выбираемых сделок или null, если он до этого не был установлен.
     *
     * @return null|string Статус выбираемых сделок
     */
    public function getStatus(): ?string
    {
        return $this->_status;
    }

    /**
     * Проверяет, был ли установлен статус выбираемых сделок.
     *
     * @return bool True если статус был установлен, false если нет
     */
    public function hasStatus(): bool
    {
        return null !== $this->_status;
    }

    /**
     * Устанавливает статус выбираемых сделок.
     *
     * @param string|null $status Статус выбираемых сделок или null, чтобы удалить значение
     *
     * @return self
     */
    public function setStatus(?string $status): self
    {
        $this->_status = $this->validatePropertyValue('_status', $status);
        return $this;
    }

    /**
     * Возвращает фильтр по описанию выбираемых сделок или null, если он до этого не был установлен.
     *
     * @return null|string Фильтр по описанию выбираемых сделок
     */
    public function getFullTextSearch(): ?string
    {
        return $this->_full_text_search;
    }

    /**
     * Проверяет, был ли установлен фильтр по описанию выбираемых сделок.
     *
     * @return bool True если фильтр по описанию был установлен, false если нет
     */
    public function hasFullTextSearch(): bool
    {
        return null !== $this->_full_text_search;
    }

    /**
     * Устанавливает фильтр по описанию выбираемых сделок.
     *
     * @param string|null $full_text_search Фильтр по описанию выбираемых сделок или null, чтобы удалить значение
     *
     * @return self
     */
    public function setFullTextSearch(?string $full_text_search): self
    {
        $this->_full_text_search = $this->validatePropertyValue('_full_text_search', $full_text_search);
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
     * Возвращает инстанс билдера объектов запросов списка сделок магазина.
     *
     * @return DealsRequestBuilder Билдер объектов запросов списка сделок
     */
    public static function builder(): DealsRequestBuilder
    {
        return new DealsRequestBuilder();
    }
}
