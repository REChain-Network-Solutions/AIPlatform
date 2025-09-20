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

/**
 * Interface RefundsRequestInterface
 *
 * @category Class
 * @package  YooKassa\Request
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $paymentId Идентификатор платежа
 * @property DateTime $createdAtGte Время создания, от (включительно)
 * @property DateTime $createdAtGt Время создания, от (не включая)
 * @property DateTime $createdAtLte Время создания, до (включительно)
 * @property DateTime $createdAtLt Время создания, до (не включая)
 * @property string $status Статус возврата
 * @property string $cursor Токен для получения следующей страницы выборки
 * @property null|int $limit Ограничение количества объектов, отображаемых на одной странице выдачи
 */
interface RefundsRequestInterface
{
    /**
     * Возвращает идентификатор платежа если он задан или null.
     *
     * @return null|string Идентификатор платежа
     */
    public function getPaymentId(): ?string;

    /**
     * Проверяет, был ли задан идентификатор платежа.
     *
     * @return bool True если идентификатор был задан, false если нет
     */
    public function hasPaymentId(): bool;

    /**
     * Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (включительно)
     */
    public function getCreatedAtGte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGte(): bool;

    /**
     * Возвращает дату создания от которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, от (не включая)
     */
    public function getCreatedAtGt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания от которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtGt(): bool;

    /**
     * Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (включительно)
     */
    public function getCreatedAtLte(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLte(): bool;

    /**
     * Возвращает дату создания до которой будут возвращены возвраты или null, если дата не была установлена.
     *
     * @return null|DateTime Время создания, до (не включая)
     */
    public function getCreatedAtLt(): ?DateTime;

    /**
     * Проверяет, была ли установлена дата создания до которой выбираются возвраты.
     *
     * @return bool True если дата была установлена, false если нет
     */
    public function hasCreatedAtLt(): bool;

    /**
     * Возвращает статус выбираемых возвратов или null, если он до этого не был установлен.
     *
     * @return null|string Статус выбираемых возвратов
     */
    public function getStatus(): ?string;

    /**
     * Проверяет, был ли установлен статус выбираемых возвратов.
     *
     * @return bool True если статус был установлен, false если нет
     */
    public function hasStatus(): bool;

    /**
     * Возвращает токен для получения следующей страницы выборки.
     *
     * @return null|string Токен для получения следующей страницы выборки
     */
    public function getCursor(): ?string;

    /**
     * Проверяет, был ли установлен токен следующей страницы.
     *
     * @return bool True если токен был установлен, false если нет
     */
    public function hasCursor(): bool;
}
