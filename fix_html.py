import re

with open("frontend/index.html", "r") as f:
    html = f.read()

# The grid container
start_marker = '<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6" id="disease-cards-grid">'
end_marker = '<div id="disease-search-empty"'

start_idx = html.find(start_marker)
end_idx = html.find(end_marker)

new_cards = """
                    <!-- Disease Card 1 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-red-500/20 dark:hover:border-red-500/10 group" data-name="panama layu fusarium">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-red-500/10 dark:bg-red-500/5 text-red-600 dark:text-red-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">🥀</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Panama Disease (Layu Fusarium)</h3>
                                <span class="text-[10px] font-bold text-red-600 dark:text-red-400 uppercase tracking-widest">Sangat Berbahaya / Lethal</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Infeksi jamur tanah ganas (Fusarium oxysporum) yang memblokir xilem akar, menyebabkan layu permanen dan kematian. Tanah dapat terinfeksi hingga puluhan tahun.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Eradikasi lahan, karantina, ganti bibit tahan varietas Cavendish.
                        </div>
                    </div>

                    <!-- Disease Card 2 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-amber-500/20 dark:hover:border-amber-500/10 group" data-name="yellow sigatoka bercak kuning">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-amber-500/10 dark:bg-amber-500/5 text-amber-600 dark:text-amber-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">🍂</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Yellow Sigatoka (Sigatoka Kuning)</h3>
                                <span class="text-[10px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest">Bahaya Sedang - Tinggi</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Bercak kuning memanjang searah urat daun yang mengganggu fotosintesis. Buah matang prematur dan ukurannya mengecil drastis.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Pangkas daun sakit, atur jarak tanam, fungisida mancozeb/chlorothalonil.
                        </div>
                    </div>

                    <!-- Disease Card 3 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-orange-500/20 dark:hover:border-orange-500/10 group" data-name="black sigatoka hitam agresif">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-orange-500/10 dark:bg-orange-500/5 text-orange-600 dark:text-orange-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">🕸️</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Black Sigatoka (Sigatoka Hitam)</h3>
                                <span class="text-[10px] font-bold text-orange-600 dark:text-orange-400 uppercase tracking-widest">Sangat Berbahaya / Lethal</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Versi jauh lebih mematikan dan agresif dari Sigatoka Kuning. Cepat mengeringkan seluruh area daun menjadi nekrotik gelap.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Fungisida sistemik rotasi ketat (triazol), perbaikan sirkulasi udara kebun.
                        </div>
                    </div>

                    <!-- Disease Card 4 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-indigo-500/20 dark:hover:border-indigo-500/10 group" data-name="bract mosaic virus mosaik seludang">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-indigo-500/10 dark:bg-indigo-500/5 text-indigo-600 dark:text-indigo-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">🧬</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Bract Mosaic Virus</h3>
                                <span class="text-[10px] font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-widest">Tinggi</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Penyakit virus yang ditularkan kutu daun, memunculkan pola mosaik khas pada seludang dan tangkai, menghambat pertumbuhan tandan buah.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Basmi vektor serangga penyebar, isolasi tanaman, gunakan kultur jaringan bebas virus.
                        </div>
                    </div>

                    <!-- Disease Card 5 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-pink-500/20 dark:hover:border-pink-500/10 group" data-name="insect pest hama serangga">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-pink-500/10 dark:bg-pink-500/5 text-pink-600 dark:text-pink-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">🐛</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Insect Pest (Hama Serangga)</h3>
                                <span class="text-[10px] font-bold text-pink-600 dark:text-pink-400 uppercase tracking-widest">Sedang</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Kerusakan fisik berupa bolong-bolong, sobekan, atau penggulungan daun akibat serangan ulat, thrips, maupun kumbang pemakan daun.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Insektisida kontak/sistemik, pembersihan gulma sarang serangga, agen hayati predator.
                        </div>
                    </div>

                    <!-- Disease Card 6 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-purple-500/20 dark:hover:border-purple-500/10 group" data-name="moko disease layu bakteri">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-purple-500/10 dark:bg-purple-500/5 text-purple-600 dark:text-purple-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">🦠</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Moko Disease (Layu Bakteri)</h3>
                                <span class="text-[10px] font-bold text-purple-600 dark:text-purple-400 uppercase tracking-widest">Sangat Berbahaya / Lethal</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Penyakit layu ganas oleh bakteri Ralstonia solanacearum. Daun layu dengan cepat dan daging buah busuk berlendir di dalam.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Eradikasi cepat tanaman, sterilisasi alat potong, hindari penanaman pada lahan basah berlebihan.
                        </div>
                    </div>

                    <!-- Disease Card 7 -->
                    <div class="disease-card glass-card border border-slate-200/60 dark:border-slate-850/60 rounded-3xl p-5 md:p-6 shadow-sm flex flex-col gap-4 transition-all duration-300 hover:-translate-y-1.5 hover:shadow-lg hover:border-emerald-500/20 dark:hover:border-emerald-500/10 group" data-name="healthy sehat aman">
                        <div class="flex items-center gap-3">
                            <div class="w-12 h-12 rounded-xl bg-emerald-500/10 dark:bg-emerald-500/5 text-emerald-600 dark:text-emerald-500 flex items-center justify-center text-2xl shrink-0 group-hover:scale-110 transition-transform">✨</div>
                            <div>
                                <h3 class="text-base font-display font-extrabold text-slate-900 dark:text-white">Healthy (Daun Sehat)</h3>
                                <span class="text-[10px] font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest">Normal</span>
                            </div>
                        </div>
                        <p class="text-xs text-slate-550 dark:text-slate-400 leading-relaxed">Tanaman berada dalam kondisi prima. Struktur sel klorofil utuh tanpa tanda-tanda invasi patogen maupun kerusakan struktur daun eksternal.</p>
                        <div class="mt-auto pt-4 border-t border-slate-200/60 dark:border-slate-800/60 text-[11px] font-medium text-slate-500">
                            <strong>Penanganan Utama:</strong> Pertahankan siklus pemupukan makro/mikro standar dan jaga kebersihan kebun berkala.
                        </div>
                    </div>
                </div>
                
                """

new_html = html[:start_idx + len(start_marker)] + "\n" + new_cards + html[end_idx:]
with open("frontend/index.html", "w") as f:
    f.write(new_html)

print("Updated frontend/index.html")
