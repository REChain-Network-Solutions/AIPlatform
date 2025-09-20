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

/**
 * Interface DealsRequestInterface.
 *
 * @category Interface
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 * @property null|string $cursor Страница выдачи результатов, которую необходимо отобразить
 * @property null|int $limit Ограничение количества объектов платежа, отображаемых на одной странице выдачи
 * @property null|DateTime $createdAtGte Время создания, от (включительно)
 * @property null|DateTime $createdAtGt Время создания, от (не включая)
 * @property null|DateTime $createdAtLte Время создания, до (включительно)
 * @property null|DateTime $createdAtLt Время создания, до (не включая)
 * @property null|DateTime $expiresAtGte Время автоматического закрытия, от (включительно)
 * @property null|DateTime $expiresAtGt Время автоматического закрытия, от (не включая)
 * @property null|DateTime $expiresAtLte Время автоматического закрытия, до (включительно)
 * @property null|DateTime $expiresAtLt Время автоматического закрытия, до (не включая)
 * @property null|string $full_text_search Фильтр по описанию сделки — параметру description
 * @property null|string $status Статус платежа
 */
interface DealsRequestInterface
{
    /**
     * Возвращает страницу выдачи результатов или null, если она до этого не была установлена.
     *
     * @return null|string Страница выдачи результатов
     */
    public function getCursor(): ?string;

    /**
     * Проверяет, была ли установлена страница выдачи результатов.
     *
     * @return bool True если страница выдачи результатов была установлена, false если нет
     */
    public function hasCursor(): bool;

    /**
     * Устанавливает страницу выдачи результатов.
     *
     * @param string $cursor Страница
     */
    public function setCursor(string $cursor): self;

    /**
     * Возвращает дату создания от которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCreatedAtGte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGte(): bool;

    /**
     * Устанавливает дату создания от которой выбираются сделки.
     *
     * @param DateTime|string|null $created_at_gte Дата
     */
    public function setCreatedAtGte(DateTime|string|null $created_at_gte): self;

    /**
     * Возвращает дату создания от которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCreatedAtGt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGt(): bool;

    /**
     * Устанавливает дату создания от которой выбираются сделки.
     *
     * @param DateTime|string|null $created_at_gt Дата
     */
    public function setCreatedAtGt(DateTime|string|null $created_at_gt): self;

    /**
     * Возвращает дату создания до которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCreatedAtLte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLte(): bool;

    /**
     * Устанавливает дату создания до которой выбираются сделки.
     *
     * @param DateTime|string|null $created_at_lte Дата
     */
    public function setCreatedAtLte(DateTime|string|null $created_at_lte): self;

    /**
     * Возвращает дату создания до которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCreatedAtLt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLt(): bool;

    /**
     * Устанавливает дату создания до которой выбираются сделки.
     *
     * @param DateTime|string|null $created_at_lt Дата
     */
    public function setCreatedAtLt(DateTime|string|null $created_at_lt): self;

    /**
     * Возвращает дату автоматического закрытия от которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, от (включительно)
     */
    public function getExpiresAtGte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата автоматического закрытия от которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtGte(): bool;

    /**
     * Устанавливает дату автоматического закрытия от которой выбираются сделки.
     *
     * @param DateTime|string|null $expires_at_gte Дата
     */
    public function setExpiresAtGte(DateTime|string|null $expires_at_gte): self;

    /**
     * Возвращает дату автоматического закрытия от которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, от (не включая)
     */
    public function getExpiresAtGt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата автоматического закрытия от которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtGt(): bool;

    /**
     * Устанавливает дату автоматического закрытия от которой выбираются сделки.
     *
     * @param DateTime|string|null $expires_at_lt Дата автоматического закрытия
     */
    public function setExpiresAtGt(DateTime|string|null $expires_at_lt): self;

    /**
     * Возвращает дату автоматического закрытия до которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, до (включительно)
     */
    public function getExpiresAtLte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата автоматического закрытия до которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtLte(): bool;

    /**
     * Устанавливает дату автоматического закрытия до которой выбираются сделки.
     *
     * @param DateTime|string|null $expires_at_lte Дата автоматического закрытия
     */
    public function setExpiresAtLte(DateTime|string|null $expires_at_lte): self;

    /**
     * Возвращает дату автоматического закрытия до которой будут возвращены сделки или null, если дата не была установлена.
     *
     * @return null|DateTime Время автоматического закрытия, до (не включая)
     */
    public function getExpiresAtLt(): ?DateTime;

    /**
     * Проверяет, была ли установлена автоматического закрытия до которой выбираются сделки.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasExpiresAtLt(): bool;

    /**
     * Устанавливает дату автоматического закрытия до которой выбираются сделки.
     *
     * @param DateTime|string|null|null $expires_at_lt Дата автоматического закрытия
     */
    public function setExpiresAtLt(DateTime|string|null $expires_at_lt): self;

    /**
     * Возвращает ограничение количества объектов сделок или null, если оно до этого не было установлено.
     *
     * @return null|int Ограничение количества объектов сделок
     */
    public function getLimit(): ?int;

    /**
     * Проверяет, было ли установлено ограничение количества объектов сделок.
     *
     * @return bool True если ограничение количества объектов сделок было установлено, false если нет
     */
    public function hasLimit(): bool;

    /**
     * Устанавливает ограничение количества объектов сделок.
     *
     * @param int|null $limit Количества объектов сделок на странице
     */
    public function setLimit(?int $limit): self;

    /**
     * Возвращает статус выбираемых сделок или null, если он до этого не был установлен.
     *
     * @return null|string Статус выбираемых сделок
     */
    public function getStatus(): ?string;

    /**
     * Проверяет, был ли установлен статус выбираемых сделок.
     *
     * @return bool True если статус был установлен, false если нет
     */
    public function hasStatus(): bool;

    /**
     * Устанавливает статус выбираемых сделок.
     *
     * @param string|null $status Статус сделок
     */
    public function setStatus(?string $status): self;

    /**
     * Возвращает фильтр по описанию сделки или null, если он до этого не был установлен.
     *
     * @return null|string Фильтр по описанию сделки
     */
    public function getFullTextSearch(): ?string;

    /**
     * Проверяет, был ли установлен фильтр по описанию сделки.
     *
     * @return bool True если фильтр по описанию сделки был установлен, false если нет
     */
    public function hasFullTextSearch(): bool;

    /**
     * Устанавливает фильтр по описанию сделки.
     *
     * @param string|null $full_text_search Фильтр по описанию сделки
     */
    public function setFullTextSearch(?string $full_text_search): self;
}
